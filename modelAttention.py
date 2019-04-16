import torch
import torch.distributed.deprecated as dist
from dataSource import Mnist, Mnist_noniid, Cifar10, Cifar10_noniid
from neuralModels import CNNMnist
import copy
from torch.multiprocessing import Process
import argparse
import time
import sys
import os
sys.stdout.flush()

def readParameters(model,name):
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint['state'])
    return model

if __name__=="__main__":
    m1=readParameters(model = CNNMnist(),name='client1.t7')
    m2=readParameters(model = CNNMnist(),name='client2.t7')
    para=m1.parameters()
