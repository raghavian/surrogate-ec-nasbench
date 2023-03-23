import json
import numpy as np
import torch
import pdb

DFILE='/home/raghav/erda/projects/surrogate_ofa/results.json'

with open(DFILE) as jfile:
    data = json.load(jfile)

N = len(data)
d = len(data[0]['d']) 
e = len(data[0]['e']) 
w = len(data[0]['w']) 
D = d+e+w+2
ofaX = torch.zeros((N,D))
ofaY = torch.zeros(N)

print('Found dataset with %d entries with %d features'%(N,D))
for i in range(N):
#    pdb.set_trace()
    ofaX[i,:d] = torch.Tensor(np.array(data[i]['d'],dtype=float)).reshape(-1)
    ofaX[i,d:d+e] = torch.Tensor(np.array(data[i]['e'],dtype=float)).reshape(-1)
    ofaX[i,d+e:d+e+w] = torch.Tensor(np.array(data[i]['w'],dtype=float)).reshape(-1)
    ofaX[i,-2] = torch.Tensor([float(data[i]['flops'])])
    ofaX[i,-1] = torch.Tensor([float(data[i]['params'])])

    ofaY[i] = torch.Tensor([float(data[i]['energy'])])

torch.save((ofaX,ofaY),'ofa_dataset.pt')


