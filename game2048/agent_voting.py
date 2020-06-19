import numpy as np
from sklearn.externals import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import time
import pandas as pd
import csv
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[.5], std=[.5])  
transform = transforms.Compose([transforms.ToTensor(), normalize])

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=True):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction

#model1
class Vgg10Conv(nn.Module):
    """
    vgg16 convolution network architecture
    """

    def __init__(self, num_cls=4, init_weights=False):
        """
        Input
            b x 1 x 4 x 4
        """
        super(Vgg10Conv, self).__init__()

        self.num_cls = num_cls
    
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(1, 64, 3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), 
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(),   
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),  
            nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  
            nn.MaxPool2d(2, stride=2, return_indices=True))

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),   nn.ReLU(),  nn.Dropout(),
            nn.Linear(512, 512),   nn.ReLU(),  nn.Dropout(),
            nn.Linear(512, num_cls)
        )

        # index of conv
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        # feature maps
        self.feature_maps = OrderedDict()
        # switch
        self.pool_locs = OrderedDict()
        # initial weight
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """
        initial weights from preptrained model by vgg16
        """
        vgg16_pretrained = models.vgg16(pretrained=True)
        # fine-tune Conv2d
        for idx, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[idx].weight.data = layer.weight.data
                self.features[idx].bias.data = layer.bias.data
        # fine-tune Linear
        for idx, layer in enumerate(vgg16_pretrained.classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data
    
    def check(self):
        model = models.vgg16(pretrained=True)
        return model

    def forward(self, x):
        """
        x.shape:        (B,C, H, W)
        return.shape:   (B , D)
        """
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

class MyAgent(nn.Module):
    '''
    input shape B x C x H x W
    output shape 1
    '''
    def __init__(self,batch_size=128):

        super(MyAgent,self).__init__()
        self.vgg10_bn = Vgg10Conv()
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size

    def loss(self,x,label):
        x = self.vgg10_bn(x)
        loss = self.criterion(x,label)
        return loss
    
    def forward(self,x):
        x = self.vgg10_bn(x)
        
        return x

#model2
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 2))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2))
        self.conv6 = nn.Conv2d(128,128,kernel_size=(2,2))
        self.fc1 = nn.Linear(128 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 4)
        self.batch_norm1 = nn.BatchNorm1d(128 * 4 * 4)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.batch_norm3 = nn.BatchNorm1d(512)

        self.initialize()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.batch_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm3(x)
        x = self.fc3(x)

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')

#model3
class Resblock(nn.Module):
    def __init__(self, channel_num):
        super(Resblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channel_num),
            nn.ReLU(),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channel_num)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.relu(out)
        return out


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 2))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2))
        self.residual = self._make_layer(Resblock,128,4)
        self.fc1 = nn.Linear(128 * 5 * 5, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 4)
        self.batch_norm1 = nn.BatchNorm1d(128 * 5 * 5)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.batch_norm3 = nn.BatchNorm1d(512)

        self.initialize()

    def _make_layer(self, block , channel_num , res_num):
        layers = []
        for i in range(res_num):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.residual(x)
        x = x.view(-1, 128 * 5 * 5)
        x = self.batch_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm3(x)
        x = self.fc3(x)

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
#model5
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 1), padding=(2, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 2))
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(2, 2))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2))
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(4, 4), padding=(2, 2))
        self.conv7 = nn.Conv2d(256, 512, kernel_size=(5, 5), padding=(2, 2))

        self.fc1 = nn.Linear(512 * 6 * 6, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 4)
        self.batch_norm1 = nn.BatchNorm1d(512 * 6 * 6)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.batch_norm3 = nn.BatchNorm1d(512)

        self.initialize()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = x.view(-1, 512 * 6 * 6)
        x = self.batch_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm3(x)
        x = self.fc3(x)

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
                
class CNN_voting(Agent):
    def __init__(self, game, display=None):
        self.game =game
        self.display=display
        self.model1 = MyAgent()
        self.model2 = Net()
        self.model3 = Net2()
        self.model4 = Net2()
        self.model5 = Net3()
        self.model1.load_state_dict(torch.load('./game2048/model1_pretrain_params.pkl'))
        self.model2.load_state_dict(torch.load('./game2048/model2_pretrain_params.pkl'))
        self.model3.load_state_dict(torch.load('./game2048/model3_pretrain_params.pkl'))
        self.model4.load_state_dict(torch.load('./game2048/model3_pretrain_params_2.pkl'))
        self.model5.load_state_dict(torch.load('./game2048/model5_pretrain_params.pkl'))
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)

    def step(self):
        present_board = self.game.board
        board = present_board.reshape(1, 1, 4, 4)
        board[np.where(board == 0)] = 1
        board = np.log2(board)
        board = torch.FloatTensor(board)
        board = Variable(board)
        
        game_board = self.game.board
        game_board = np.expand_dims(game_board,axis = 2)
        game_board = transform(game_board).float()
        game_board = game_board.reshape(1,1,4,4)
        
        directions = []
        self.search_func1 = self.model1
        self.search_func2 = self.model2
        self.search_func3 = self.model3
        self.search_func4 = self.model4
        self.search_func5 = self.model5

        self.search_func1.eval()
        self.search_func2.eval()
        self.search_func3.eval()
        self.search_func4.eval()
        self.search_func5.eval()
        
        output1 = self.search_func1(game_board)
        output2 = self.search_func2(board)
        output3 = self.search_func3(board)
        output4 = self.search_func4(board)
        output5 = self.search_func5(board)
        direction1 = output1.data.max(1, keepdim=True)[1].item()
        directions.append(direction1)
        direction2 = output2.data.max(1, keepdim=True)[1].item()
        directions.append(direction2)
        direction3 = output3.data.max(1, keepdim=True)[1].item()
        directions.append(direction3)
        direction4 = output4.data.max(1, keepdim=True)[1].item()
        directions.append(direction4)
        direction5 = output5.data.max(1, keepdim=True)[1].item()
        directions.append(direction5)
        
        '''
        if direction1 == direction2:
            direction = direction1
        elif direction1 == direction3:
            direction = direction3
        elif direction2 == direction3:
            direction = direction2
        else:
            direction = direction1
        '''
        directions_array = np.array(directions)
        directions = np.unique(directions_array)
        direction_now = -1
        count_now = -1
        print(directions_array)
        for i in range(len(directions)):
            count = np.sum(directions_array==directions[i])
            if count > count_now:
                count_now = count
                direction_now = directions[i]
        return int(direction_now)