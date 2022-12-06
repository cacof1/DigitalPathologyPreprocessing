import sys
sys.path.insert(0,'/home/cacof1/Software/DigitalPathologyPreprocessing/')


from Dataloader.Dataloader import *
from Utils.PreprocessingTools import Preprocessor
import toml
from Utils import GetInfo
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from torchvision import transforms
import torch
from QA.StainNormalization import ColourNorm
from Model.ConvNet import ConvNet

pd.set_option('display.max_rows', None) 
n_gpus = torch.cuda.device_count()  # could go into config file
config = toml.load(sys.argv[1])

########################################################################################################################
# 1. Download all relevant files based on the configuration file

SVS_dataset = QueryROI(config)
## For hierarchical sorting
SVS_dataset['ROIName'] = pd.Categorical(SVS_dataset.ROIName, ordered=True, categories=config['CRITERIA']['ROI'])
SVS_dataset = SVS_dataset.sort_values('ROIName')

SVS_dataset = SVS_dataset.reset_index()

SynchronizeSVS(config, SVS_dataset)


########################################################################################################################
# 2. Pre-processing: create tile_dataset from annotations

preprocessor = Preprocessor(config)
tile_dataset = preprocessor.getTilesFromAnnotations(SVS_dataset)
config['DATA']['N_Classes'] = len(tile_dataset[config['DATA']['Label']].unique())


########################################################################################################################
# 3. Model

# Set up logging, model checkpoint
name = GetInfo.format_model_name(config)
if 'logger_folder' in config['CHECKPOINT']:
    logger = TensorBoardLogger(os.path.join('lightning_logs', config['CHECKPOINT']['logger_folder']), name=name)
else:
    logger = TensorBoardLogger('lightning_logs', name=name)

lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(dirpath=config['CHECKPOINT']['Model_Save_Path'],
                                      monitor=config['CHECKPOINT']['Monitor'],
                                      filename=name + '-epoch{epoch:02d}-' + config['CHECKPOINT']['Monitor'] + '{' +
                                      config['CHECKPOINT']['Monitor'] + ':.2f}',
                                      save_top_k=1,
                                      mode=config['CHECKPOINT']['Mode'])

pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)
                   
# transforms: augment data on training set
if config['AUGMENTATION']['Rand_Operations'] > 0:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config[
            'NORMALIZATION'] else None,
        transforms.ToPILImage(),
        transforms.RandAugment(num_ops=config['AUGMENTATION']['Rand_Operations'],
                               magnitude=config['AUGMENTATION']['Rand_Magnitude']),
        # this only operates on 8-bit images (not normalised float32 tensors)
        transforms.ToTensor(),  # this also normalizes to [0,1].,
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

else:
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # this also normalizes to [0,1].,
        ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config[
            'NORMALIZATION'] else None,
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# transforms: colour norm only on validation set
val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    ColourNorm.Macenko(saved_fit_file=config['NORMALIZATION']['Colour_Norm_File']) if 'Colour_Norm_File' in config[
        'NORMALIZATION'] else None,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create LabelEncoder
le = preprocessing.LabelEncoder()
le.fit(tile_dataset[config['DATA']['Label']])

# Load model and train
trainer = pl.Trainer(gpus=n_gpus,
                     strategy='ddp',
                     benchmark=True,
                     max_epochs=config['ADVANCEDMODEL']['Max_Epochs'],
                     precision=config['BASEMODEL']['Precision'],
                     callbacks=[checkpoint_callback, lr_monitor],
                     logger=logger)

model = ConvNet(config, label_encoder=le)
########################################################################################################################
# 3. Dataloader

data = DataModule(
    tile_dataset,
    batch_size=config['BASEMODEL']['Batch_Size'],
    train_transform=train_transform,
    val_transform=val_transform,
    train_size=config['DATA']['Train_Size'],
    val_size=config['DATA']['Val_Size'],
    inference=False,
    dim_list=config['BASEMODEL']['Patch_Size'],
    vis_list=config['BASEMODEL']['Vis'],
    n_per_sample=config['DATA']['N_Per_Sample'],
    target=config['DATA']['Label'],
    sampling_scheme=config['DATA']['Sampling_Scheme'],
    svs_folder=config['DATA']['SVS_Folder'],
    label_encoder=le
)

# Give the user some insight on the data
GetInfo.ShowTrainValTestInfo(data, config)

# Load model and train
trainer.fit(model, data)

