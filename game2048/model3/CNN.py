import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np
import csv

batch_size = 128
NUM_EPOCHS = 30

# load data
csv_data = pd.read_csv('data.csv')
csv_data = csv_data.values
X0 = np.int64(csv_data)
X0 = np.reshape(X0, (-1,4,4))
direction_data = pd.read_csv('direction.csv')
direction_data = direction_data.values
Y0 = np.int64(direction_data)

csv_data1 = pd.read_csv('data3.csv')
csv_data1 = csv_data1.values
X1 = np.int64(csv_data1)
X1 = np.reshape(X1, (-1,4,4))
direction_data1 = pd.read_csv('direction3.csv')
direction_data1 = direction_data1.values
Y1 = np.int64(direction_data1)

csv_data2 = pd.read_csv('data4.csv')
csv_data2 = csv_data2.values
X2 = np.int64(csv_data2)
X2 = np.reshape(X2, (-1,4,4))
direction_data2 = pd.read_csv('direction4.csv')
direction_data2 = direction_data2.values
Y2 = np.int64(direction_data2)
'''
csv_data3 = pd.read_csv('data3.csv')
csv_data3 = csv_data3.values
X3 = np.int64(csv_data3)
X3 = np.reshape(X3, (-1,4,4))
direction_data3 = pd.read_csv('direction3.csv')
direction_data3 = direction_data3.values
Y3 = np.int64(direction_data3)
'''

print(X0.shape)
print(X1.shape)
print(X2.shape)
print(Y0.shape)
print(Y1.shape)
print(Y2.shape)

X = np.vstack([X0, X1, X2])
Y = np.vstack([Y0, Y1, Y2])
print(X.shape)
print(Y.shape)
X[np.where(X == 0)] = 1
X = np.log2(X)

#writer = SummaryWriter('./runs/batch_size_{}_epoch{}'.format(batch_size, NUM_EPOCHS))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,shuffle=False)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

train_dataset = torch.utils.data.TensorDataset(X_train,Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test,Y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True
)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False
)

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

model = Net()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        data = data.unsqueeze(dim=1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model.state_dict(), 'model_saved/params2_epoch_{}.pkl'.format(epoch))


def test(epoch):
    test_loss = 0
    correct = 0
    # model.load_state_dict(torch.load('model_saved/params_epoch_{}.pkl'.format(epoch)))
    for data, target in test_loader:
        #data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        with torch.no_grad():
            data,target = Variable(data).cuda(), Variable(target).cuda()  
        data = data.unsqueeze(dim=1)
        output = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(output, target.squeeze(), size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    #writer.add_scalar('Test_loss', test_loss, epoch)
    #writer.add_scalar('Test_accuracy', 100. * float(correct) / len(test_loader.dataset), epoch)


for epoch in range(0, NUM_EPOCHS):
    train(epoch)
    test(epoch)
