import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Sequential
import pdb
import numpy as np
from PyTorch_BayesianCNN.utils.BBBlayers import BBBConv2d

def weight_init_fn(m):
    #nn.init.xavier_normal_(m.weight.data)
    nn.init.kaiming_normal_(m.weight.data)
    #nn.init.normal_(m.weight.data)

def bias_init_fn(m):
    nn.init.constant_(m.bias.data, 0)


def conv3x3(inplanes, planes, stride):
    return nn.Conv2d(inputs=inplanes, outputs=planes, stride=stride)

class ResnetBuildingBlock(nn.Module):
    #expansion = 1

    def __init__(self, inplanes, planes, stride=(1,1)):
        super(ResnetBuildingBlock, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = BBBConv2d(inplanes, planes, kernel_size=3, stride=stride, padding = 1)
        #self.conv1.apply(weight_init_fn)
        #self.conv1.apply(bias_init_fn)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = BBBConv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1)
        #self.conv2.apply(weight_init_fn)
        #self.conv2.apply(bias_init_fn)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        x.shape
        residual = x
        # first convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        # add residual
        out += residual
        out = self.relu(out)
        return out


class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    #def __init__(self):
    #    self.x = 0
    
    def reshape(self,x):
        n = x.size(0)
        m = x.size(1)
        x = x.view(n,m)
        return x

    def __call__(self,x):
        return self.reshape(x)


class all_cnn_module(nn.Module):
    """
    Create a nn.Sequential model containing all of the layers of the All-CNN-C as specified in the paper.
    https://arxiv.org/pdf/1412.6806.pdf
    Use a AvgPool2d to pool and then your Flatten layer as your final layers.
    You should have a total of exactly 23 layers of types:
    - nn.Dropout
    - nn.Conv2d
    - nn.ReLU
    - nn.AvgPool2d
    - Flatten
    :return: a nn.Sequential model
    """
    def __init__(self):
        super(all_cnn_module, self).__init__()
        self.Sequential = [[] for i in range(11)]
        self.Sequential[0] =   BBBConv2d(1,64,(3,3),padding=0, stride=(2,2))
        #self.Sequential[0].apply(weight_init_fn)
        self.Sequential[1] =   nn.ELU()
        self.Sequential[2] =   ResnetBuildingBlock(64,64,(1,1))
        self.Sequential[3] =   BBBConv2d(64,128,(3,3),padding=0, stride=(2,2))
        #self.Sequential[3].apply(weight_init_fn)
        self.Sequential[4] =   nn.ELU()
        self.Sequential[5] =   ResnetBuildingBlock(128,128,(1,1))
        self.Sequential[6] =   BBBConv2d(128,512,(3,3),padding=0, stride=(2,2))
        #self.Sequential[6].apply(weight_init_fn)
        self.Sequential[7] =   nn.ELU()
        self.Sequential[8] =   ResnetBuildingBlock(512,512,(1,1))
        self.Sequential[9] =   nn.AvgPool2d((249,1))
        self.Sequential[10] =   Flatten()

        self.net = nn.Sequential(*self.Sequential)
        self.classifier = nn.Linear(512,11)
        self.classifier.apply(weight_init_fn)
        self.classifier.apply(bias_init_fn)

    def forward(self,X, is_embedding=False):
        E = self.net.forward(X)
        E_norm = 16*(E/torch.norm(E))
        if is_embedding:
            return E_norm
        else:
            param = self.classifier.named_parameters()
            for name, params in param:
                if name in ['weight']:
                    weights = params
            return self.classifier(E_norm)/torch.norm(weights,2,1).reshape(1,len(weights))












#A = Variable(torch.FloatTensor([[[[1]], [[2]], [[3]], [[4]]],
#     [[[1]], [[2]], [[3]], [[4]]],
#     [[[1]], [[2]], [[3]], [[4]]]]))

#print (A.size(0))
#print A.shape[0], A.shape[1], A.shape[2], A.shape[3]



#flatten = Flatten()

#print (flatten(A.view(3,4,1,1))) 

#print (all_cnn_module())


#X = torch.cuda.FloatTensor(np.random.randint(0,100,(1,1,2000,20)))

#cnn = all_cnn_module()
#cnn.cuda()
#print (cnn.net[0])

#print (cnn.net.forward(X).shape)
#print (cnn.forward(X).shape)
#print (torch.norm(cnn.forward(X),2,0).shape)

