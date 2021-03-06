# Copyright (C) 2020 RADTorch and Mohamed Elbanan, MD
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

## Code Last Updated/Checked: 08/01/2020


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch, torchvision, datetime, time, pickle, pydicom, os, math, random, itertools, ntpath, copy, xmltodict, subprocess

import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets

from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from pathlib import Path
from datetime import datetime
from sklearn import metrics, tree
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, SGDClassifier, ElasticNet
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.feature_selection import SelectKBest, chi2,  f_classif, mutual_info_classif, RFECV, RFE, VarianceThreshold
from xgboost import XGBClassifier
from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from collections import Counter
from IPython.display import display
from bokeh.io import output_notebook, show
from math import pi
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter, Tabs, Panel, ColumnDataSource, Legend
from bokeh.plotting import figure, show
from bokeh.layouts import row, gridplot, column
from bokeh.transform import factor_cmap, cumsum
from bokeh.palettes import viridis, Paired, inferno, brewer, d3
from statistics import mode, mean
from efficientnet_pytorch import EfficientNet

# import streamlit as st



# GENERAL
global log_dir
logfile='/content/log.text'
log_dir = '/'

# VIS
TOOLS="hover,save,box_zoom,reset,wheel_zoom, box_select"
COLORS3=["#d11141","#00b159","#00aedb","#f37735","#ffc425","#cccccc","#8c8c8c","#cccccc", "#ffc425","#f37735","#00aedb","#00b159"]
COLORS2=['#1C1533', '#3C6FAA', '#10D8B8', '#FBD704', '#FF7300','#F82716','#FF7300', '#FBD704', '#10D8B8', '#3C6FAA']*100
COLORS=['#93D5ED', '#45A5F5', '#4285F4', '#2F5EC4', '#0D47A1','#2F5EC4', '#4285F4', '#45A5F5',]*100


# CORE
model_dict={
'vgg11':{'name':'vgg11','input_size':224, 'output_features':4096},
'vgg11_bn':{'name':'vgg11_bn','input_size':224, 'output_features':4096},
'vgg13':{'name':'vgg13','input_size':224, 'output_features':4096},
'vgg13_bn':{'name':'vgg13_bn','input_size':224, 'output_features':4096},
'vgg16':{'name':'vgg16','input_size':224, 'output_features':4096},
'vgg16_bn':{'name':'vgg16_bn','input_size':224, 'output_features':4096},
'vgg19':{'name':'vgg19','input_size':244, 'output_features':4096},
'vgg19_bn':{'name':'vgg19_bn','input_size':224, 'output_features':4096},
'resnet18':{'name':'resnet18','input_size':224, 'output_features':512},
'resnet34':{'name':'resnet34','input_size':224, 'output_features':512},
'resnet50':{'name':'resnet50','input_size':224, 'output_features':2048},
'resnet101':{'name':'resnet101','input_size':224, 'output_features':2048},
'resnet152':{'name':'resnet152','input_size':224, 'output_features':2048},
'wide_resnet50_2':{'name':'wide_resnet50_2','input_size':224, 'output_features':2048},
'wide_resnet101_2':{'name':'wide_resnet101_2','input_size':224, 'output_features':2048},
'alexnet':{'name':'alexnet','input_size':256, 'output_features':4096},
'efficientnet-b0':{'name':'efficientnet-b0','input_size':224, 'output_features':1280},
'efficientnet-b1':{'name':'efficientnet-b1','input_size':224, 'output_features':1280},
'efficientnet-b2':{'name':'efficientnet-b2','input_size':224, 'output_features':1408},
'efficientnet-b3':{'name':'efficientnet-b3','input_size':224, 'output_features':1536},
'efficientnet-b4':{'name':'efficientnet-b4','input_size':224, 'output_features':1792},
'efficientnet-b5':{'name':'efficientnet-b5','input_size':224, 'output_features':2048},
'efficientnet-b6':{'name':'efficientnet-b6','input_size':224, 'output_features':2304},
'efficientnet-b7':{'name':'efficientnet-b7','input_size':224, 'output_features':2560},
# 'inception_v3':{'name':'inception_v3','input_size':299, 'output_features':2048},

              }

supported_models=[x for x in model_dict.keys()]

supported_multi_label_image_classification_losses=[]

supported_nn_optimizers=[
'Adam',
'AdamW',
'SparseAdam',
'Adamax',
'ASGD',
'RMSprop',
'SGD']

supported_nn_loss_functions=[
'NLLLoss',
'CrossEntropyLoss',
'MSELoss',
'PoissonNLLLoss',
'BCELoss',
'BCEWithLogitsLoss',
'MultiLabelMarginLoss',
'SoftMarginLoss',
'MultiLabelSoftMarginLoss',
'CosineSimilarity',
]


SUPPORTED_CLASSIFIER=[
'nn_classifier',
'linear_regression',
'sgd',
'logistic_regression',
'ridge',
'knn',
'decision_trees',
'random_forests',
'gradient_boost',
'adaboost',
'xgboost',
]


# DATA
IMG_EXTENSIONS=(
'.jpg',
'.jpeg',
'.png',
'.ppm',
'.bmp',
'.pgm',
'.tif',
'.tiff',
'.webp')
