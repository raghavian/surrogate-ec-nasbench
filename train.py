import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.nn.functional as F

from dataset import NonGraph7V
from models import MLP

import argparse
import pdb
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def train(max_epoch=50):
    #### Training and validation loop!
    trLoss = []
    trAcc = []
    vlLoss = []
    vlAcc = []
    
    for epoch in range(max_epoch):          ### Run the model for max_epochs

        epLoss = 0
        for x, y in train_loader:       ### Fetch a batch of training inputs
            yHat = model(x)               ####### Obtain a prediction from the network
            loss = (criterion(yHat,y))    ######### Compute loss bw prediction and ground truth

            ### Backpropagation steps
            ### Clear old gradients
            optimizer.zero_grad()
            ### Compute the gradients wrt model weights
            loss.backward()
            ### Update the model parameters
            optimizer.step()

            epLoss += loss.item()

        trLoss.append(epLoss/len(train_loader))

        epLoss = 0
        for x, y in valid_loader: #### Fetch validation samples

            yHat = model(x) ##########
            loss = (criterion(yHat,y))#######

            epLoss += loss.item()
        vlLoss.append(epLoss/len(valid_loader))

        print('Epoch: %03d, Tr. Loss: %.4f, Vl.Loss: %.4f'
              %(epoch,trLoss[-1],vlLoss[-1]))
    plt.clf()
    plt.plot(trLoss,label='training')
    plt.plot(vlLoss,label='validation')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_curve.pdf', dpi=300)
    return model

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
parser.add_argument('--vary_train', action='store_true', default=False, \
        help='Train the model for different amounts of training data/ Paper plot')
parser.add_argument('--save', action='store_true', default=True, \
        help='Save the trained model')
parser.add_argument('--model', type=str, default=None, help='Pretrained model')

args = parser.parse_args()
print('Loading and preprocessing '+args.data.split('/')[-1])

if args.model == None:
    model_name = 'models/model_H_'+repr(args.hidden)+'.pt' 

### Obtain a distribution of number of events for different thresholds
num_events = []

### Make torch dataset
dataset = NonGraph7V(data_file=args.data)
#dataset = Graph7V(data_file=args.data)


#### Make training, validation and test sets
N = len(dataset)
nIp = dataset[0][0].shape[-1]
nTrain = int(0.7*N)
nValid = int(0.1*N)
nTest = N - nTrain -nValid
train_set, valid_set, test_set = random_split(dataset,[nTrain, nValid, nTest])
B = args.batch

### Wrapping the datasets with DataLoader class 
valid_loader = DataLoader(valid_set,batch_size=B, shuffle=True)
test_loader = DataLoader(test_set,batch_size=B, shuffle=True)
if args.vary_train:
    trial = 10
    num_train = 100
    nInter = nTrain//num_train
    metric = np.zeros((trial,nInter))
    xrange = (np.linspace(num_train,nTrain,nInter)).astype(int)

    for rand in range(trial):
        trIdx = np.random.permutation(nTrain)
        tIdx = 0
        for t in xrange: 
            train_set = Subset(dataset,trIdx[:t])
            print(rand,t)
            train_loader = DataLoader(train_set,batch_size=B, shuffle=True)
            print("Ntrain: %d, NValid: %d, NValid: %d"%(len(train_set),nValid,nTest))

            ### Instantiate a model! 
            model = MLP(nIp=nIp,nhid=args.hidden) 
            criterion = nn.MSELoss() ############ Loss function to be optimized. 
            optimizer = torch.optim.Adam(model.parameters(),lr=args.lr) 
            model = train(args.epochs)
            y_list = np.zeros(1)
            yHat_list = np.zeros(1)

            for x, y in test_loader: #### Fetch validation samples
                yHat = model(x) ##########
                yHat_list = np.concatenate((yHat_list,yHat.view(-1).detach().numpy()))
                y_list = np.concatenate((y_list,y.view(-1).detach().numpy()))
            metric[rand,tIdx] = np.abs((y_list[1:] - yHat_list[1:])).mean()
            tIdx += 1

    plt.clf()
    plt.plot(xrange,metric.mean(0))
    plt.fill_between(xrange,metric.mean(0)-metric.std(0),metric.mean(0)+metric.std(0),alpha=0.2)
    plt.xlabel('# of training datapoints')
    plt.ylabel('Mean Absolute Error on fixed test set')
    plt.tight_layout()
    plt.savefig('surrogate_ntrain.pdf',dpi=300)
    pdb.set_trace()
    np.save('metrics.npy',metrics)
else:

    ### Standard model training with all training data 
    train_loader = DataLoader(train_set,batch_size=B, shuffle=True)

    ### Train the neural network!
    print("Retraining from scratch...")
    model = train(args.epochs)
    args.model = model_name

    ### Save the trained model
    if args.save:
        torch.save(model.state_dict(), model_name)

    ### Visualize predictions 
    plt.clf()

    y_list = np.zeros(1)
    yHat_list = np.zeros(1)
    for x, y in train_loader: #### Fetch validation samples
        yHat = model(x) ##########
        yHat_list = np.concatenate((yHat_list,yHat.view(-1).detach().numpy()))
        y_list = np.concatenate((y_list,y.view(-1).detach().numpy()))

    #pdb.set_trace()
    plt.scatter(yHat_list[1:],y_list[1:],marker='x',s=2)
    y_list = np.zeros(1)
    yHat_list = np.zeros(1)


    for x, y in test_loader: #### Fetch validation samples
        yHat = model(x) ##########
        yHat_list = np.concatenate((yHat_list,yHat.view(-1).detach().numpy()))
        y_list = np.concatenate((y_list,y.view(-1).detach().numpy()))

    plt.scatter(yHat_list[1:],y_list[1:],marker='+')

    ymax = int(y_list.max() + 1)
    plt.plot(np.arange(ymax),np.arange(ymax))

    plt.savefig('scatter.pdf',dpi=200)
