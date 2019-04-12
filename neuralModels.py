import torch
import time
import torch.nn as nn
import torch.nn.functional as F

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()               # input shape(1,28,28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)     #output_shape (32,24,24)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)    #output_shape (64,20,20)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))      # output_shape(32,12,12)
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))      #output_shape(64,4,4)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])  # output_shape(1,64*4*4)，view修改张量形状而不修改数据
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
