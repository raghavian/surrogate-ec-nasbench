import torch
import pdb
from sklearn.metrics.pairwise import pairwise_distances
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np

operations = np.array(['input','conv1x1-bn-relu','conv3x3-bn-relu','maxpool3x3','output'])

def op2vec(string):
    N = len(string)
    vec = np.zeros(N,dtype=int)
    string = np.array(string)
    for sIdx in range(N):
        vec[sIdx] = np.where(operations == string[sIdx])[0]
    return vec + 1

class OFADataset(Dataset):
    def __init__(self, data_file = '/home/raghav/Dropbox/playground/python/projects/surrogate_ec_nas/ofa_dataset.pt'):
        super().__init__()

        self.data, self.energy = torch.load(data_file)
#        pdb.set_trace()
#        self.data = self.data[self.energy < 5e-4] 
#        self.energy = self.energy[self.energy < 5e-4]
        N = len(self.energy)

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, index):

        return self.data[index], self.energy[index]*1e3


class SurrogateDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        
        adj_full = data['adjacency_matrix']
        nodes = data['module_operations']
        self.nodes = op2vec(nodes)
#        self.energy = data['energy (kWh)']
        self.param = data['params']

        N = len(self.nodes)
        idx = len(np.triu_indices(7)[0])
        adj = torch.zeros((N,idx))
        self.adj = torch.FloatTensor(adj_full[np.triu_indices(7)]).reshape(-1,1)

        self.nodes = torch.FloatTensor(self.nodes).reshape(-1,1)
#        self.energy = torch.FloatTensor([self.energy]).reshape(-1,1)
        self.param = torch.FloatTensor([self.param]).reshape(-1,1)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, index):
        feature = torch.cat((self.adj,self.nodes,self.param/1e6),dim=0).reshape(-1)

        return feature #, self.energy*1000


class NonGraph7V(Dataset):
    def __init__(self, data_file = '/home/raghav/Dropbox/playground/python/projects/surrogate_ec_nas/7v_data.pt'):
        super().__init__()

        adj_full, self.nodes, self.energy, self.param = torch.load(data_file)
        N = len(self.nodes)
        idx = len(np.triu_indices(7)[0])
        adj = torch.zeros((N,idx))
        for i in range(len(self.nodes)):
            adj[i] = adj_full[i][np.triu_indices(7)]

        self.adj = adj
        self.nodes = self.nodes.type(torch.FloatTensor)#.reshape#(-1,1)
        self.energy = self.energy.type(torch.FloatTensor).reshape(-1,1)
        self.param = self.param.type(torch.FloatTensor).reshape(-1,1)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, index):
        feature = torch.cat((self.adj[index],self.nodes[index],self.param[index]/1e6),dim=0).reshape(-1)
#        feature = torch.cat((self.adj[index],self.nodes[index]),dim=0).reshape(1,-1)

#        feature = self.param[index]/1e6


        return feature, self.energy[index]*1000

class Graph7V(Dataset):
    def __init__(self, data_file = '/home/raghav/Dropbox/playground/python/projects/surrogate_ec_nas/7v_data.pt'):
        super().__init__()

        self.adj, self.nodes, self.energy, self.param = torch.load(data_file)
        N = len(self.nodes)
        idx = len(np.triu_indices(7)[0])

        self.nodes = self.nodes.type(torch.FloatTensor)
        self.energy = self.energy.type(torch.FloatTensor).reshape(-1,1)
        self.param = self.param.type(torch.FloatTensor).reshape(-1,1)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, index):

#        feature = torch.cat((self.adj[index],self.nodes[index],self.param[index]/1e6),dim=0).reshape(1,-1)
#        feature = torch.cat((self.adj[index],self.nodes[index]),dim=0).reshape(1,-1)

        return self.adj[index], self.nodes[index], self.param[index], self.energy[index]*1000

