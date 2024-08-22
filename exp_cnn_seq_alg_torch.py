import os
import pandas as pd
import numpy as np
from joblib import dump

import math
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# data_tf = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.ToTensor( ),# 将原有数据转化成张量图像，值在(0,1)
#         torchvision.transforms.Normalize([0.5],[0.5])# 将数据归一化到(-1,1)，参数（均值，标准差）。
#     ]
# )
# # data = mnist.read_image_file("D:/jinyfeng/datas/MNIST/MNIST/raw/train-images-idx3-ubyte") # torch.Size([60000, 28, 28])
# # print(data.shape)
# root_path = 'D:/jinyfeng/datas/MNIST'
# #训练和测试集预处理
# train_dataset = datasets.MNIST(root=root_path, train=True, transform=data_tf, download=True)
# test_dataset = datasets.MNIST(root=root_path, train=False, transform=data_tf)
# #加载数据集
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# for data in train_loader:
#     img, label = data
#     # print('img.shape========', img.shape)
#     print('label========', label)
#     # img = img.view(img.size(0), -1)
#     # print('img.shape========', img.shape)


class SelfDataset(Dataset):  # 定义一个自己的数据集
    def __init__(self, filepath):
        data = pd.read_excel(filepath)
        # x_data = data[['WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY',
        #           'WEIGHT_45', 'WEIGHT_LOCATION_45', 'UX_45', 'UY_45', 'ROTX_45', 'ROTY_45',
        #           'WEIGHT_90', 'WEIGHT_LOCATION_90', 'UX_90', 'UY_90', 'ROTX_90', 'ROTY_90',
        #           'WEIGHT_135', 'WEIGHT_LOCATION_135', 'UX_135', 'UY_135', 'ROTX_135', 'ROTY_135',
        #           'WEIGHT_180', 'WEIGHT_LOCATION_180', 'UX_180', 'UY_180', 'ROTX_180', 'ROTY_180',
        #           'WEIGHT_225', 'WEIGHT_LOCATION_225', 'UX_225', 'UY_225', 'ROTX_225', 'ROTY_225',
        #           'WEIGHT_270', 'WEIGHT_LOCATION_270', 'UX_270', 'UY_270', 'ROTX_270', 'ROTY_270',
        #           'WEIGHT_315', 'WEIGHT_LOCATION_315', 'UX_315', 'UY_315', 'ROTX_315', 'ROTY_315']]
        x_data = data[['SECTION_NUMBER', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY',
                 'SECTION_NUMBER_45', 'WEIGHT_45', 'WEIGHT_LOCATION_45', 'UX_45', 'UY_45', 'ROTX_45', 'ROTY_45',
                 'SECTION_NUMBER_90', 'WEIGHT_90', 'WEIGHT_LOCATION_90', 'UX_90', 'UY_90', 'ROTX_90', 'ROTY_90',
                 'SECTION_NUMBER_135', 'WEIGHT_135', 'WEIGHT_LOCATION_135', 'UX_135', 'UY_135', 'ROTX_135', 'ROTY_135',
                 'SECTION_NUMBER_180', 'WEIGHT_180', 'WEIGHT_LOCATION_180', 'UX_180', 'UY_180', 'ROTX_180', 'ROTY_180',
                 'SECTION_NUMBER_225', 'WEIGHT_225', 'WEIGHT_LOCATION_225', 'UX_225', 'UY_225', 'ROTX_225', 'ROTY_225',
                 'SECTION_NUMBER_270', 'WEIGHT_270', 'WEIGHT_LOCATION_270', 'UX_270', 'UY_270', 'ROTX_270', 'ROTY_270',
                 'SECTION_NUMBER_315', 'WEIGHT_315', 'WEIGHT_LOCATION_315', 'UX_315', 'UY_315', 'ROTX_315', 'ROTY_315']]
        y_data = data['DAMAGE_LOCATION']
        # Standardize the features

        # scaler = StandardScaler()
        # scaler.fit(x_data)

        scaler = StandardScaler()
        # scaler = MinMaxScaler(feature_range=(-1,1))
        # scaler.fit(x_data)

        x_scaled = scaler.fit_transform(x_data)
        # X_scaled = scaler.transform(X)
        # print(x_scaled)

        # reshaped_x = np.array(x_data).reshape(x_data.shape[0], 8, 6)
        reshaped_x = np.array(x_scaled).reshape(x_scaled.shape[0], 8, 7)

        print('reshaped_x.shape=======', reshaped_x.shape)
        y_data = np.array(y_data)
        # print(y_data.shape)
        self.len = x_data.shape[0]
        self.x_data = torch.from_numpy(reshaped_x.astype(np.float32))
        self.y_data = torch.from_numpy(y_data)
        print(type(self.x_data), type(self.y_data))
        print((self.x_data).dtype, (self.y_data).dtype)
        print(self.x_data.shape, self.y_data.shape)

    def __getitem__(self, index):  # __getitem__为必须需要实现的一个方法 用于从索引取出数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # __getitem__为必须需要实现的一个方法
        return self.len


batch_size=16
# trainfile_path = 'D:/jinyfeng/datas/jinyfeng(2)/train_val_seq_v1.xls'
trainfile_path = 'D:/jinyfeng/datas/jinyfeng(2)/train_val_seq_v2_train.xls'
testfile_path = 'D:/jinyfeng/datas/jinyfeng(2)/train_val_seq_v2_test.xls'
train_dataset = SelfDataset(trainfile_path)
test_dataset = SelfDataset(testfile_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Batch_Net(nn.Module):
    """
    在上面的Activation_Net的基础上，增加了一个加快收敛速度的方法——批标准化
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
    # def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(Batch_Net, self).__init__()
        
        # self.layer1 = nn.Linear(in_dim, n_hidden_1)
        # self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        # self.layer3 = nn.Linear(n_hidden_2, out_dim)

        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        # self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        
        # self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        # self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        # self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        return x

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        layer1 = nn.Sequential()
        # layer1.add_module('conv1', nn.Conv2d(1, 16, 3, padding=1))
        layer1.add_module('conv1', nn.Conv2d(1, 32, 3))
        layer1.add_module('bn1', nn.BatchNorm2d(32))
        layer1.add_module('relu1', nn.ReLU())
        # layer1.add_module('pool1', nn.MaxPool2d(2, 2))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        # layer2.add_module('conv2', nn.Conv2d(16, 32, 3, padding=1))
        layer2.add_module('conv2', nn.Conv2d(32, 64, 3))
        layer2.add_module('bn2', nn.BatchNorm2d(64))
        layer2.add_module('relu2', nn.ReLU())
        # layer2.add_module('pool2', nn.MaxPool2d(2, 2))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc1', nn.Linear(64*4*3, 256))
        layer3.add_module('fc2', nn.Linear(256, 19))
        # layer3.add_module('fc3', nn.Linear(84, 19))
        self.layer3 = layer3

    def forward(self, x):
        # print('x.shape======', x.shape)
        x = self.layer1(x)
        # print('x.shape======', x.shape)
        x = self.layer2(x)
        # print('x.shape======', x.shape)
        x = x.view(x.size(0), -1)
        # print('x.shape======', x.shape)
        x = self.layer3(x)
        # print('x.shape======', x.shape)
        return x


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        # 卷积层
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,1), stride=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4,1), stride=1, padding=0),
        )

        # 全连接层
        self.fullconn_layer = nn.Sequential(
            # input:torch.Size([batch_size, 16*5*5])
            nn.Linear(32 * 7, 120),
            # nn.Sigmoid(),
            nn.ReLU(),
            # input:torch.Size([batch_size, 120])
            nn.Linear(120, 84),
            # nn.Sigmoid(),
            nn.ReLU(),
            # input:torch.Size([batch_size, 84])
            nn.Linear(84, 19),
        )
        # output:torch.Size([10, 10])

    def forward(self, x):
        output = self.conv_layer(x)  # output:torch.Size([batch_size, 16, 5, 5])
        # output = output.view(batch_size,-1)   # output:torch.Size([10, 16*5*5])
        output = output.view(x.size(0), -1)
        output = self.fullconn_layer(output)  # output:torch.Size([10, 10])
        return output


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device=====', device)
# model.to(device)


# model = Lenet()
model = LeNet5()
# model = Batch_Net(768, 300, 100, 10) # 28*28=768
# model = Batch_Net(48, 128, 64, 19)
# model = Batch_Net(56, 128, 64, 19)
# model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# model.loss_func = torch.nn.BCELoss()
# model.metric_func = lambda y_pred, y_true: roc_auc_score(y_true.data.numpy(), y_pred.data.numpy())
# model.metric_name = "auc"

if torch.cuda.is_available():
    # model = model.cuda()
    model.to(device)

learning_rate=0.001
criterion = nn.CrossEntropyLoss() #softmax与交叉熵一起
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))

# criterion = nn.BCELoss()  # 二分类的交叉熵 size_average表示对交叉熵求均值


epochs=50
iter = 0
for epoch in range(epochs):
    for data in train_loader:
        model.train()
        # 1. Prepare data
        inputs, labels = data  # data是由__getitem__返回的一个元组？ (inputs,labels)
        # print('inputs.shape=======', inputs.shape, labels.shape)
        # inputs = inputs.view(inputs.size(0), -1)
        # print(inputs.shape)

        inputs=inputs.unsqueeze(1)
        # print('inputs.shape=======', inputs.shape)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            # inputs.to(device)
            # labels.to(device)
        # else:
        #     inputs = inputs
        #     labels = labels

        # 2. Forward
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        
        # 3. Backward
        optimizer.zero_grad()
        loss.backward()
        # 4. Update
        optimizer.step()

        iter+=1
        #每迭代50次打印一次
        if iter%50 == 0:
            # print(epoch, loss.item())
            print('epoch: {}, iter:{}, loss: {:.4}'.format(epoch, iter, loss.data.item()))

            #模型评估
            print('Start eval!')
            model.eval()
            eval_loss = 0
            eval_acc = 0
            for data in test_loader:
                img, label = data
                # img = img.view(img.size(0), -1)
                img=img.unsqueeze(1)
                if torch.cuda.is_available():
                    img = img.cuda()
                    label = label.cuda()
                    # img.to(device)
                    # label.to(device)

                out = model(img)
                loss = criterion(out, label)
                eval_loss += loss.data.item()*label.size(0)
                _, pred = torch.max(out, 1) #onehout编码，dim=1选取最大值
                num_correct = (pred == label).sum()
                eval_acc += num_correct.item()

            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))



import datetime

metric_name = model.metric_name
# 用于记录训练过程中的loss和metric
dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])
print("Start Training...")
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("==========" * 8 + "%s" % nowtime)

for epoch in range(1, epochs + 1):

    # 1，训练循环-------------------------------------------------
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1

    for step, (features, labels) in enumerate(dl_train, 1):
        # train模块，也可以直接放在这里

        loss, metric = train(model, features, labels)

        # 打印batch级别日志
        loss_sum += loss
        metric_sum += metric
        # 设置打印freq
        if step % log_step_freq == 0:
            print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                  (step, loss_sum / step, metric_sum / step))

    # 2，验证循环-------------------------------------------------
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1

    for val_step, (features, labels) in enumerate(dl_valid, 1):
        # valid模块
        val_loss, val_metric = valid(model, features, labels)

        val_loss_sum += val_loss
        val_metric_sum += val_metric

    # 3，记录日志-------------------------------------------------
    info = (epoch, loss_sum / step, metric_sum / step,
            val_loss_sum / val_step, val_metric_sum / val_step)
    dfhistory.loc[epoch - 1] = info

    # 打印epoch级别日志
    print(("\nEPOCH = %d, loss = %.3f," + metric_name + \
           "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
          % info)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)

print('Finished Training...')
