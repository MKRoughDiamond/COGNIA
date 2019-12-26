import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self,IMG_SIZE=28):
        super(MNISTCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,8,5,stride=2)
        self.BN1 = torch.nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,8,5,stride=2)
        self.BN2 = torch.nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8,8,3,stride=1)
        self.fc = nn.Linear(8*2*2,10)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.BN1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.BN2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1,8*2*2)
        x = self.fc(x)
        x = torch.softmax(x,dim=-1)
        return x
