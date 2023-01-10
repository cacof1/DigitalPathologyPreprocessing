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
import matplotlib.pyplot as plt
#pd.set_option('display.max_rows', None)
config_file  = sys.argv[1]
config       = toml.load(config_file)

########################################################################################################################
# 1. Download all relevant ROI based on the configuration file
SVS_dataset = QueryROI(config)

## For hierarchical contours -- accelerate preprocessing 
SVS_dataset['ROIName'] = pd.Categorical(SVS_dataset.ROIName, ordered=True, categories=config['CRITERIA']['ROI'])
SVS_dataset = SVS_dataset.sort_values('ROIName')
SVS_dataset = SVS_dataset.reset_index()

## Download SVS
SynchronizeSVS(config, SVS_dataset)

########################################################################################################################
# 3. Pre-processing: create tile_dataset from annotations list
preprocessor = Preprocessor(config)
tile_dataset = preprocessor.getTilesFromAnnotations(SVS_dataset)


## Manual fiddling
tile_dataset.loc[ tile_dataset['tissue_type'].str.contains('Artifact'),'tissue_type'] = 'Artifact'
tile_dataset.loc[ tile_dataset['tissue_type'].str.contains('Muscle'),'tissue_type'] = 'Muscle'

config['DATA']['N_Classes'] = len(tile_dataset[config['DATA']['Label']].unique())
print(tile_dataset, config['DATA']['N_Classes'])

print(tile_dataset['tissue_type'].value_counts())

# Set up logging, model checkpoint
name = GetInfo.format_model_name(config)
if 'logger_folder' in config['CHECKPOINT']:
    logger = TensorBoardLogger(os.path.join('lightning_logs', config['CHECKPOINT']['logger_folder']), name=name)
else:
    logger = TensorBoardLogger('lightning_logs', name=name)

########################################################################################################################
# 4. Model
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir,
                                      monitor=config['CHECKPOINT']['Monitor'],
                                      filename='checkpoint-epoch{epoch:02d}-' + config['CHECKPOINT']['Monitor'] + '{' + config['CHECKPOINT']['Monitor'] + ':.2f}', 
                                      save_top_k=1,
                                      mode=config['CHECKPOINT']['Mode'])

pl.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)                   
# transforms: augment data on training set
if config['AUGMENTATION']['Rand_Operations'] > 0:
    train_transform = transforms.Compose([
        transforms.ToTensor(), ## Convert to [0,1] by dividing by 255
        ColourNorm.Macenko(),
        transforms.ToPILImage(),
        transforms.RandAugment(num_ops=config['AUGMENTATION']['Rand_Operations'],
                               magnitude=config['AUGMENTATION']['Rand_Magnitude']),
        transforms.ToTensor(), ## Convert to [0,1] by dividing by 255
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

else:
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # this also normalizes to [0,1].,
        ColourNorm.Macenko(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# transforms: colour norm only on validation set
val_transform = transforms.Compose([
    transforms.ToTensor(),  # this also normalizes to [0,1].
    ColourNorm.Macenko(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create LabelEncoder
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(tile_dataset[config['DATA']['Label']])

# Load model and train
print("N GPUs: ",torch.cuda.device_count())
trainer = pl.Trainer(gpus=torch.cuda.device_count(),  # could go into config file
                     strategy='bagua',
                     benchmark=True,
                     max_epochs=config['ADVANCEDMODEL']['Max_Epochs'],
                     precision=config['BASEMODEL']['Precision'],
                     callbacks=[checkpoint_callback, lr_monitor],
                     logger=logger)

model = ConvNet(config, label_encoder=label_encoder)
########################################################################################################################
# 5. Dataloader
data = DataModule(
    tile_dataset,
    batch_size=config['BASEMODEL']['Batch_Size'],
    train_transform=train_transform,
    val_transform=val_transform,
    train_size=config['DATA']['Train_Size'],
    val_size=config['DATA']['Val_Size'],
    inference=False,
    n_per_sample=config['DATA']['N_Per_Sample'],
    target=config['DATA']['Label'],
    sampling_scheme=config['DATA']['Sampling_Scheme'],
    label_encoder=label_encoder
)

# Give the user some insight on the data
GetInfo.ShowTrainValTestInfo(data, config, label_encoder)

# Load model and train/validate
trainer.fit(model, data)

## Test
trainer.test(model, data.test_dataloader())

## Write config file in logging folder for safekeeping
with open(logger.log_dir+"/Config.ini", "w+") as toml_file:
    toml.dump(config, toml_file)
    toml_file.write("Train transform:\n")
    toml_file.write(str(train_transform))
    toml_file.write("Val/Test transform:\n")
    toml_file.write(str(val_transform))

