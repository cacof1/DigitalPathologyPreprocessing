from Dataloader.Dataloader import *
from Utils.PreprocessingTools import Preprocessor
import toml
from torch import cuda
from Utils import GetInfo
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
import pytorch_lightning as pl
from Utils import MultiGPUTools
from torchvision import transforms
import torch
from QA.StainNormalization import ColourAugment
from Model.ConvNet import ConvNet
import datetime
import multiprocessing as mp

config = toml.load(sys.argv[1])
n_gpus = 1 #cuda.device_count()


########################################################################################################################
# 1. Download all relevant ROI based on the configuration file
SVS_dataset = QueryImageFromCriteria(config).reset_index()
SynchronizeSVS(config, SVS_dataset)

########################################################################################################################
# 2. Pre-processing: create tile_dataset from annotations list

preprocessor = Preprocessor(config)
tile_dataset = preprocessor.getAllTiles(SVS_dataset, background_fraction_threshold=0.7)

########################################################################################################################
# 3. Model

# Data transformation

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Pad tile_dataset such that the final batch size can be divided by n_gpus.
n_pad = MultiGPUTools.pad_size(len(tile_dataset), n_gpus, config['BASEMODEL']['Batch_Size'])
tile_dataset = MultiGPUTools.pad_dataframe(tile_dataset, n_pad) 

data = DataLoader(DataGenerator(tile_dataset, transform=val_transform, target=config['DATA']['Label'], inference=True),
                  batch_size=config['BASEMODEL']['Batch_Size'],
                  num_workers=int(.8 * mp.Pool()._processes),
                  persistent_workers=True,
                  shuffle=False,
                  pin_memory=True)

trainer = pl.Trainer(devices=n_gpus,
                     accelerator="gpu",
                     strategy=pl.strategies.DDPStrategy(timeout=datetime.timedelta(seconds=10800)),
                     benchmark=False,
                     precision=config['BASEMODEL']['Precision'],
                     callbacks=[TQDMProgressBar(refresh_rate=1)])

model = ConvNet.load_from_checkpoint(config['CHECKPOINT']['Model_Save_Path'])
model.eval()

########################################################################################################################
# 4. Predict

predictions = trainer.predict(model, data)
ordered_preds = MultiGPUTools.reorder_predictions(predictions)  # reorder if processing was done on multiple GPUs
predicted_classes_prob = torch.Tensor.cpu(torch.cat(ordered_preds))

# Drop padding
tile_dataset           = tile_dataset.iloc[:-n_pad]
predicted_classes_prob = predicted_classes_prob[:-n_pad]


########################################################################################################################
# 5. Save (separate from Omero upload in case it crashes)

# Start by saving the dataframe in case there is a failure in the code below - by experience on multiple
# gpus the code hangs here sometimes...
tile_dataset = tile_dataset.drop(columns="SVS_PATH")  # drop SVS path before saving.
tile_dataset.to_csv('full_tile_dataset_after_inference.csv')

tissue_names = model.LabelEncoder.inverse_transform(np.arange(predicted_classes_prob.shape[1]))
for tissue_no, tissue_name in enumerate(tissue_names):
    
    tile_dataset['prob_' + config['DATA']['Label'] + '_' + tissue_name] = predicted_classes_prob[:, tissue_no]
    tile_dataset = tile_dataset.fillna(0)

for id_external, df_split in tile_dataset.groupby(tile_dataset.id_external):
    npy_file = SaveFileParameter(config, df_split, str(id_external))

print('Npy export complete - now uploading onto Omero...')






