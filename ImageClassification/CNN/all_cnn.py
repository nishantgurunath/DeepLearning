import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Sequential


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


def all_cnn_module():
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
    return Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(3,96,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(96,96,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(96,96,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(96,192,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(192,192,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(192,192,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv2d(192,192,3,padding=0),
            nn.ReLU(),
            nn.Conv2d(192,192,1,padding=0),
            nn.ReLU(),
            nn.Conv2d(192,10,1,padding=0),
            nn.ReLU(),
            nn.AvgPool2d(6),
            Flatten()
            )



A = Variable(torch.FloatTensor([[[[1]], [[2]], [[3]], [[4]]],
     [[[1]], [[2]], [[3]], [[4]]],
     [[[1]], [[2]], [[3]], [[4]]]]))

#print (A.size(0))
#print A.shape[0], A.shape[1], A.shape[2], A.shape[3]



flatten = Flatten()

#print (flatten(A.view(3,4,1,1))) 

#print (all_cnn_module())
