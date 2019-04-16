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

# parameter settings
LR = 0.001
MAX_ROUND = 100
ROUND_NUMBER_FOR_SAVE = 10
ROUND_NUMBER_FOR_REDUCE = 5
CLIENT_NUMBER=10
IID = True
DATA_SET = 'Mnist'
MODEL = 'CNN'

def get_local_data(size, rank, batchsize):
    if IID == True:
        if DATA_SET == 'Mnist':
            train_loader = Mnist(batchsize).get_train_data()
        if DATA_SET == 'Cifar10':
            train_loader = Cifar10(batchsize).get_train_data()
    else:

        if DATA_SET == 'Mnist':
            train_loader = Mnist_noniid(batchsize, size).get_train_data(rank)
        if DATA_SET == 'Cifar10':
            train_loader = Cifar10_noniid(batchsize, size).get_train_data(rank)

    return train_loader

def get_testset():
    if IID == True:
        if DATA_SET == 'Mnist':
            test_loader = Mnist().get_test_data()
        if DATA_SET == 'Cifar10':
            test_loader = Cifar10().get_test_data()
    else:
        if DATA_SET == 'Mnist':
            test_loader = Mnist_noniid().get_test_data()
        if DATA_SET == 'Cifar10':
            test_loader = Cifar10_noniid().get_test_data()
    for step, (b_x, b_y) in enumerate(test_loader):
        test_x = b_x
        test_y = b_y
    return test_x, test_y

def load_model(model, rank):
    print('===> Try resume from checkpoint')
    if os.path.exists('autoencoder' + str(rank) + '.t7'):
        checkpoint = torch.load('autoencoder' + str(rank) + '.t7')
        model.load_state_dict(checkpoint['state'])
        round = checkpoint['round']
        print('===> Load last checkpoint data')
    else:
        round = 0
    return model, round

def save_model(model, round, rank):
    print('===> Saving models...')
    state = {
        'state': model.state_dict(),
        'round': round,
        }
    torch.save(state, 'client5' +'.t7')


def run(size, rank, epoch, batchsize):
    # 确定数据集和模型
    if MODEL == 'CNN' and DATA_SET == 'Mnist':
        model = CNNMnist()
    # 设定优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_func = torch.nn.CrossEntropyLoss()
    # load训练数据
    train_loader = get_local_data(size, rank, batchsize)
    if rank == 0 :                                             # rank=0就是master节点，只需要对master节点用测试集
        test_x, test_y = get_testset()
    model, round = load_model(model, rank)
    while round < MAX_ROUND:
        sys.stdout.flush()
        if rank == 0:
            test_output = model(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Round: ', round, ' Rank: ', rank, '| test accuracy: %.2f' % accuracy)

        # 训练过程
        for epoch_cnt in range(epoch):
            for step, (b_x, b_y) in enumerate(train_loader): #以batchsize为步长批训练数据，这里是100个一批-
                optimizer.zero_grad()             # 把梯度置零
                output = model(b_x)             #输入+模型=输出
                loss = loss_func(output, b_y)              #输出+标答=loss
                loss.backward()                #反向传播
                optimizer.step()               #更新参数
        round+=1
    save_model(model=model, round=1,rank=rank)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', '-s', type=int, default=1)                #5个用户
    parser.add_argument('--epoch', '-e', type=int, default=1)               #本地就训练1个epoch
    parser.add_argument('--batchsize', '-b', type=int, default=100)         #批训练数目是100
    args = parser.parse_args()

    size = args.size
    epoch = args.epoch
    batchsize = args.batchsize
    for rank in range(0, size):
        run(size=1, rank=rank, epoch=1, batchsize=100)
