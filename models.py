import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

### CNN Based AE
### Autoencoder class
class CNN_AE(nn.Module):
    def __init__(self,nCh,nIp,nhid,latent_dim,kernel=15):
        super(CNN_AE, self).__init__()
        pad = kernel // 2
        ### Encoder layers 3
        self.enc1 = nn.Conv1d(nCh,nhid,kernel_size=kernel,padding=pad)
        self.enc2 = nn.Conv1d(nhid,nhid,kernel_size=kernel,padding=pad)
        self.enc3 = nn.Conv1d(nhid,nhid,kernel_size=kernel,padding=pad)
        self.enc4 = nn.Linear(nhid*nIp//8,latent_dim)
        ### Decoder 3 layers
        self.dec0 = nn.Linear(latent_dim,nhid*nIp)
        self.dec1 = nn.Conv1d(nhid,nhid,kernel_size=kernel,padding=pad)
        self.dec2 = nn.Conv1d(nhid,nhid,kernel_size=kernel,padding=pad)
        self.dec3 = nn.Conv1d(nhid,nCh,kernel_size=kernel,padding=pad)
        self.nhid = nhid
        self.avg = nn.AvgPool1d(2)
    def encode(self, x):

        ### Fill in the encoder
        ### by calling the corresponding layers
        ### initialized above.
        ### You can use F.relu() to call the
        ### rectified linear unit activation function
        x = F.elu(self.enc1(x))
        x = self.avg(x)
        x = F.elu(self.enc2(x))
        x = self.avg(x)
        x = F.elu(self.enc3(x))
        x = self.avg(x)
        z = self.enc4(x.reshape(x.shape[0],1,-1))

        return z

    def decode(self, z):

        ### Fill in the decoder
        ### by calling the corresponding layers
        ### initialized above.
        ### You can use torch.sigmoid() to call the
        ### sigmoid activation function
        x = F.elu(self.dec0(z))
        x = F.elu(self.dec1(x.reshape(x.shape[0],self.nhid,-1)))
        x = F.elu(self.dec2(x))
        xHat = torch.sigmoid(self.dec3(x))

        return xHat

    def forward(self, x):
        ### Autoencoder returns the reconstruction
        ### and latent representation
        z = self.encode(x)
        # decode z
        xHat = self.decode(z)
        return xHat,z


### MLP based AE
### Autoencoder class
class MLP(nn.Module):
    def __init__(self,nIp,nhid=32,op_dim=1):
        super(MLP, self).__init__()
        
        ### Encoder layers 3
        self.fc1 = nn.Linear(nIp,nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.fc2 = nn.Linear(nhid,nhid // 2)
        self.bn2 = nn.BatchNorm1d(nhid//2)
        self.fc3 = nn.Linear(nhid // 2,nhid // 4)
        self.bn3 = nn.BatchNorm1d(nhid//4)
        self.fc4 = nn.Linear(nhid // 4,nhid // 4)
        self.bn4 = nn.BatchNorm1d(nhid//4)
        self.fc5 = nn.Linear(nhid // 4 , op_dim)

    def forward(self, x):
#        pdb.set_trace()
#        xip = x[:,[-1]]
        ### Fill in the encoder
        ### by calling the corresponding layers
        ### initialized above.
        ### You can use F.relu() to call the
        ### rectified linear unit activation function
        x = F.gelu(self.fc1(x))
        # x = self.bne1(x)
        x = F.gelu(self.fc2(x))
        # x = self.bne2(x)
        x = F.gelu(self.fc3(x))
        # x = self.bne3(x)
        x = F.gelu(self.fc4(x))
        # x = self.bne4(x)
        yHat = self.fc5(x)

        return yHat

