import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import glob
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import pickle as pickle

#Using MedMnist as dataset, link: https://github.com/MedMNIST/MedMNIST
import medmnist
from medmnist import INFO, Evaluator
import logging
import sys

import tensorflow as tf
from tensorflow.keras import datasets, layers
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
#from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D, GlobalAveragePooling2D, Input, Conv3D, MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from keras.applications.densenet import DenseNet121
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn import metrics
from keras.optimizers.schedules import ExponentialDecay
from keras.initializers import Constant
from tensorflow.python.keras import backend as K

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import TensorDataset
import torchvision.models as models

import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import LearningRateFinder,BatchSizeFinder, EarlyStopping


##PREPROCESSING

import nibabel as nib

from scipy import ndimage


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 32
    desired_width = 64
    desired_height = 64
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

#LOAD EXISTING SCANS

abnormal_scans = np.load('./Abnormal and Normal Scans NP arrays/abnormal_scans_MOREDATA.npy')
normal_scans = np.load('./Abnormal and Normal Scans NP arrays/normal_scans_MOREDATA.npy')

abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

#SPLIT DATA
X_train = np.concatenate((abnormal_scans, normal_scans), axis = 0)
y_train = np.concatenate((abnormal_labels, normal_labels), axis = 0)
y_train = to_categorical(y_train, 2)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2], X_train.shape[3])
X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1], X_val.shape[2], X_val.shape[3])

#Convert to RGB: 

X_train = torch.cuda.FloatTensor(X_train)
y_train = torch.cuda.FloatTensor(y_train)
X_val = torch.cuda.FloatTensor(X_val)
y_val = torch.cuda.FloatTensor(y_val) 

X_train = X_train.expand(-1, 3, -1, -1, -1)
X_val = X_val.expand(-1, 3, -1, -1, -1)

print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')
#DEFINE LOADER, SPLIT INTO BATCHES

BATCH_SIZE = 32

dataset_train = TensorDataset(X_train, y_train)
dataloader_train = DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle = True)

dataset_test = TensorDataset(X_val, y_val)
dataloader_test = DataLoader(dataset_test, batch_size = BATCH_SIZE, shuffle = False)

#PYTORCH & DEFINE MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#PICK WHICH MODEL TO RUN

# model = ResNet18()

#===========================
criterion = nn.CrossEntropyLoss()

#optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)


class LitModel(pl.LightningModule):

    # def DenseNet(): 
    #     model = monai.networks.nets.DenseNet121(spatial_dims = 3, in_channels = 1, out_channels = 2).to(device)
    #     return model

    def __init__(self, num_classes, learning_rate):
        super().__init__()
        self.ResNet = models.video.r3d_18(weights = 'DEFAULT')
        #self.DenseNet = monai.networks.nets.DenseNet121(spatial_dims = 3, in_channels = 1, out_channels = 2)

        #first_conv_layer = nn.Conv3d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        num_features = self.ResNet.fc.in_features
        self.ResNet.fc = nn.Linear(num_features, num_classes)
        nn.init.xavier_uniform_(self.ResNet.fc.weight)
        nn.init.zeros_(self.ResNet.fc.bias)
        #self.ResNet = nn.Sequential(first_conv_layer, self.ResNet)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = learning_rate

    def model(self):
        model = self.ResNet
        return model
    
    def forward(self, x):
        x = self.ResNet.forward(x)
        return x

    def training_step(model, dataloader_train):
        X_batch, y_batch = dataloader_train
        y_pred = model.forward(X_batch)


        loss = criterion(y_pred, torch.max(y_batch, 1)[1])

        accuracy = BinaryAccuracy().to(device)
        accuracy = accuracy(y_pred, y_batch)

        model.log_dict({'train_loss' : loss, 'train_acc' : accuracy}, prog_bar = True, on_step = False, on_epoch = True)
        return loss
    
    def validation_step(model, dataloader_test):
        val_X_batch, val_y_batch = dataloader_test
        val_pred = model.forward(X_val)

        val_loss = criterion(val_pred, y_val)
        
        val_accuracy = BinaryAccuracy().to(device)
        val_accuracy = val_accuracy(val_pred, y_val)

        model.log_dict({'val_loss' : val_loss, 'val_acc' : val_accuracy}, prog_bar = True, on_step = False, on_epoch = True)

        return val_loss

    def configure_optimizers(model):
        return torch.optim.Adam(model.parameters(), lr = model.lr)

model = LitModel(num_classes = 2, learning_rate = 0.001).to(device)
MAX_lr = 0.01
MIN_lr = 0.0000001
#callbacks = [LearningRateFinder(MIN_lr, MAX_lr, num_training_steps = 10, mode = 'linear', update_attr = True, attr_name = 'lr')]
logger = CSVLogger('logs', name = 'exp')

early_stop_callback = EarlyStopping(monitor = 'val_acc', patience = 5, verbose = True, mode = 'max')
trainer = pl.Trainer(min_epochs = 0, max_epochs = 50, log_every_n_steps = 12, precision = 16, logger = logger, callbacks = [early_stop_callback])

tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model, dataloader_train, dataloader_test)
new_lr = lr_finder.suggestion()
print(new_lr)
#lr_finder.results()
def run_model():
    trainer.fit(model, dataloader_train, dataloader_test)

run_model()

metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')

aggreg_metrics = []
agg_col = 'epoch'
for i, dfg in metrics.groupby(agg_col):
    agg = dict(dfg.mean())
    agg[agg_col] = i
    aggreg_metrics.append(agg)

df_metrics = pd.DataFrame(aggreg_metrics)
df_metrics[['train_loss', 'val_loss']].plot(
    grid = True, legend = True, xlabel = 'Epoch', ylabel = 'loss'
)
plt.savefig(f'{trainer.logger.log_dir}/loss.png')
df_metrics[['train_acc', 'val_acc']].plot(
    grid = True, legend = True, xlabel = 'Epoch', ylabel = 'acc'
)
plt.savefig(f'{trainer.logger.log_dir}/acc.png')

def save_model(model_name): 
    torch.save(model, model_name + '.pt')

