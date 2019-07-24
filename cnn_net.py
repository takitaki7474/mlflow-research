import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_out):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(None, 32, 3, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 3, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 3, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 3, 1)
        self.conv5 = nn.Conv2d(32, 32, 3, 3, 1)
        self.conv6 = nn.Conv2d(32, 32, 3, 3, 1)
        self.fc1 = nn.Linear(None, 512)
        self.fc2 = nn.Linear(None, n_out)

    def forword(self, x):
        h = F.relu(self.conv1(x))
        h = nn.MaxPool2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.conv3(x))
        h = nn.MaxPool2d(F.relu(self.conv4(h)), 2)
        h = F.relu(self.conv5(x))
        h = nn.MaxPool2d(F.relu(self.conv6(h)), 2)
        h = F.dropout(F.relu(self.fc1(h)))
        h = self.fc2(h)
        return h
        
