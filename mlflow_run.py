import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cnn_net

def train():

    net = cnn_net.Net()
