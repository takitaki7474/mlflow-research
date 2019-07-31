import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cnn_net

def train(epoch, trainloader):

    net = cnn_net.Net(3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for e in range(epoch):

        running_loss = 0.0
        for i, data in enumerate(trainloader):

            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            print("gggggggggggggg",outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print("loss: {0}".format(running_loss/2000))
                running_loss = 0.0

    print("Finished Training")
