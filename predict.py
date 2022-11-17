import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import sys

from dataset import NonGraph7V, SurrogateDataset
from models import MLP

import argparse
import pdb
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


### Main starts here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Spike data',\
        default='/home/raghav/Dropbox/playground/python/projects/surrogate_ec_nas/7v_data.pt')
parser.add_argument('--output', type=str, default=None,help='Output filename')
parser.add_argument('--epochs', type=int, default=100,help='No. of hidden units at input in AE')
parser.add_argument('--hidden', type=int, default=128,help='No. of hidden units at input in AE')
parser.add_argument('--batch', type=int, default=32,help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-4,help='Learning rate')

parser.add_argument('--save', action='store_true', default=True, \
        help='Save the trained model')
parser.add_argument('--model', type=str, default=None, help='Pretrained model')

args = parser.parse_args()
print('Loading and preprocessing '+args.data.split('/')[-1])

if args.model == None:
    model_name = 'models/model_H_'+repr(args.hidden)+'.pt' 

### Instantiate a model! 
data_file='/home/raghav/dsl/projects/EC-NAS-Bench/data/7V_4epochs.pkl'
obj = open(data_file,"rb")
pdb.set_trace()
### Make torch dataset
nIp = dataset[0][0].shape[-1]
model = MLP(nIp=36,nhid=128) 
### Load pretrained model
if args.model is not None:
    model.load_state_dict(torch.load(args.model,map_location=device))
    print("Using pretrained model...")
else:
    print('No pretrained model found!')
    exit(1)
### 
### Loop over the hash and load a dictionary with features
##
for data in tf_record:
#    data = pickle.load(obj)[0]
    dataset = SurrogateDataset(data=data)
    #dataset = NonGraph7V()
    dataloader = DataLoader(dataset, batch_size=32)

    for x in dataloader: #### Fetch validation samples
        ## Predict energy
        yHat = model(x) ##########

