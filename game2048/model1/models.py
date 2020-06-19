from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from game2048.vgg import Vgg
class MyAgent(nn.Module):
    '''
    input shape B x C x H x W
    output shape 1
    '''
    def __init__(self,batch_size=128):

        super(MyAgent,self).__init__()
        self.vgg = Vgg()
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size

    def loss(self,x,label):
        x = self.vgg(x)
        loss = self.criterion(x,label)
        return loss
    
    def forward(self,x):
        x = self.vgg(x)
        return x