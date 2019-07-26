import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_out):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 3, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 3, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 3, 1)
        self.conv5 = nn.Conv2d(32, 32, 3, 3, 1)
        self.conv6 = nn.Conv2d(32, 32, 3, 3, 1)
        self.fc1 = nn.Linear(32 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, n_out)

    def forword(self, x):
        h = F.relu(self.conv1(x))
        h = nn.MaxPool2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.conv3(h))
        h = nn.MaxPool2d(F.relu(self.conv4(h)), 2)
        h = F.relu(self.conv5(h))
        h = nn.MaxPool2d(F.relu(self.conv6(h)), 2)
        h = F.dropout(F.relu(self.fc1(h)))
        h = self.fc2(h)
        return h

class Net2(nn.Module):
    # NNの各構成要素を定義
    def __init__(self):
        super(Net2, self).__init__()

        # 畳み込み層とプーリング層の要素定義
        self.conv1 = nn.Conv2d(28, 32, 3)  # (入力, 出力, 畳み込みカーネル（5*5）)
        self.pool = nn.MaxPool2d(2, 2)  # (2*2)のプーリングカーネル
        self.conv2 = nn.Conv2d(13, 32, 3)
        # 全結合層の要素定義
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # (入力, 出力)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)  # クラス数が１０なので最終出力数は10

    # この順番でNNを構成
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1->relu->pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2->relu->pool
        x = x.view(-1, 16 * 5 * 5)  # データサイズの変更
        x = F.relu(self.fc1(x))  # fc1->relu
        x = F.relu(self.fc2(x))  # fc2->relu
        x = self.fc3(x)
        return x
