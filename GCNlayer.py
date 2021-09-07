import torch
import torch.nn as nn
from pygcn.layers import GraphConvolution

class GCNlayer(nn.Module):

    def __init__(self,D_in,D_out,D_hidden,M_gcn):
        super(GCNlayer, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.gcnconv1 = GraphConvolution(D_in, D_hidden)
        self.gcnconv2 = GraphConvolution(D_hidden, D_hidden)
        # self.gcnconv3 = GraphConvolution(D_hidden, D_out)
        # self.fc1 = nn.Linear(D_in, D_hidden)  # 5*5 from image dimension
        # self.fc2 = nn.Linear(D_hidden, D_hidden)
        self.fc3 = nn.Linear(D_hidden, D_out)
        self.m1=nn.GroupNorm(1,D_hidden)
        # self.m2=nn.GroupNorm(1,D_hidden)
        # self.m3=nn.GroupNorm(D_out,D_out)
        self.d1 = nn.Dropout(p=0.2)
        self.d2 = nn.Dropout(p=0.2)
        self.lr = nn.LeakyReLU(0.01)
        self.M_gcn_p=M_gcn
        
    def forward(self, ddata):
        x,adjx=ddata[0],ddata[1]
        
        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        # x = self.lr(z_score_normalize_tensor_(self.fc1(x)))
        # x=self.lr(self.m1(self.fc1(x)))
        # x=self.lr(self.m2(self.fc2(x)))
        x = self.d1(self.lr(self.m1(self.gcnconv1(x, adjx))))

        x = self.d2(self.lr(self.m1(self.gcnconv2(x, adjx))))
        # x = F.softmax(self.fc3(x),dim = 1)
        x = self.fc3(x)
        
        # x=F.relu(data_minmax_tensor_(self.fc1(x)))
        # x=F.relu(data_minmax_tensor_(self.fc2(x)))
        # x=data_minmax_tensor_(self.fc3(x))
        return x