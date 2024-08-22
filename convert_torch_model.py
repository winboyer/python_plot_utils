import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import joblib
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DomainTransformer(nn.Module):
    def __init__(self, input_dim, hidden_num, num_classes1, num_classes2):
        super(DomainTransformer, self).__init__()
        self.less_hidnum = 2
        self.domain_trans_in = nn.Sequential(
            nn.Linear(input_dim-self.less_hidnum, hidden_num),
            nn.ReLU(),
        )
        self.domain_trans_out = nn.Sequential(
            nn.Linear(hidden_num, input_dim-self.less_hidnum),
            # nn.ReLU(),
        )
        self.pretrain_model = TransformerClassifier(input_dim, num_classes1, num_classes2)
        for p in self.pretrain_model.parameters():
            p.requires_grad=False
    def forward(self, x):
        #取出需要转换空间映射输入
        # print('x.shape=========', x.shape)
        #拆分
        x = x[:, self.less_hidnum:]
        x_cond = x[:, :self.less_hidnum]
        # print('x.shape=========', x.shape)
        x = self.domain_trans_in(x)
        x = self.domain_trans_out(x)
        # print('x.shape=========', x.shape)
        #合并
        x = torch.cat((x_cond, x), 1)
        # print('x.shape=========', x.shape)
        x1, x2 = self.pretrain_model(x)
        return x1, x2

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes1, num_classes2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=4), num_layers=2)
        self.fc1 = nn.Linear(128, num_classes1)
        self.fc2 = nn.Linear(128, num_classes2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2

# Train the MLP model
# Initialize the MLPClassifier with a random state for reproducibility
activate_layer=nn.ReLU()
# activate_layer=nn.Sigmoid()
class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super(MLP,self).__init__()    # 
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(8,80)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(80,40)  # 第二个隐含层
        for p in self.parameters():
            p.requires_grad=False
        self.fc3 = torch.nn.Linear(40,19)   # 输出层
        self.fc4 = torch.nn.Linear(40,4)   # 输出层
        
    def forward(self,din):
        # 前向传播， 输入值：din, 返回值 dout
        din = din.view(-1, 8)       # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        dout = F.relu(self.fc2(dout))
        dout1 = self.fc3(dout)
        dout2 = self.fc4(dout)

        # dout1 = F.softmax(self.fc3(dout), dim=1)  # 输出层使用 softmax 激活函数
        # dout2 = F.softmax(self.fc4(dout), dim=1)  # 输出层使用 softmax 激活函数
        return dout1, dout2

def init_weights(model_param):
    if type(model_param) == nn.Linear or type(model_param) == nn.Conv2d:
        # nn.init.xavier_uniform_(model_param.weight)
        nn.init.kaiming_normal_(model_param.weight)


def convert_baseline_model(file_name):
    print('file_name=======',file_name)
    # model_params = torch.load(file_name)
    model_params = torch.load(file_name, map_location=torch.device('cpu'))
    # print('model_params========', model_params)
    
    key_mapping = {
        "embedding": "pretrain_model.embedding",
        "transformer": "pretrain_model.transformer",
        "fc1": "pretrain_model.fc1",
        "fc2": "pretrain_model.fc2"
    }
    new_param_dict = dict()
    for old_key in list(model_params.keys()):
        parameter = model_params.pop(old_key)
        print('old_key========', old_key, type(parameter), parameter.shape)
        if old_key.startswith("transformer"):
            new_key = old_key.replace(
                "transformer",
                key_mapping["transformer"]
            )
            new_param_dict[new_key] = nn.Parameter(parameter)
        elif old_key.startswith("embedding"):
            new_key = old_key.replace(
                "embedding",
                key_mapping["embedding"]
            )
            # print('parameter=======', type(parameter), parameter)
            new_param_dict[new_key] = nn.Parameter(parameter)
        elif old_key.startswith("fc1"):
            new_key = old_key.replace(
                "fc1",
                key_mapping["fc1"]
            )
            new_param_dict[new_key] = nn.Parameter(parameter)
        elif old_key.startswith("fc2"):
            new_key = old_key.replace(
                "fc2",
                key_mapping["fc2"]
            )
            new_param_dict[new_key] = nn.Parameter(parameter)
    
    # print('new_param_dict=======', new_param_dict)
    return new_param_dict

# model = MLP()
# model = TransformerClassifier(8, 19, 4)
model = DomainTransformer(8,8,19,4)

# params = model.parameters()
# print('model=======', model)
# print('params=======', params)

model_file = 'torch_baseline_model_transformer_v1.pth'
# model_file = 'torch_baseline_model_v4.pth'
param_dicts = convert_baseline_model(model_file)

# torch.save(model.state_dict(), 'sgd_baseline_model_ft_v1.pth')#只保存模型权重参数，不保存模型结构
# model.load_state_dict(param_dicts)
model.load_state_dict(param_dicts, strict=False)
model.eval()
torch.save(model.state_dict(), 'domaintransmodel_transformer_v1_in6.pth')#保存整个model的状态
# model=torch.load(mymodel.pth)#这里已经不需要重构模型结构了，直接load就可以
# model.eval()



# torch.save(model.state_dict(), 'sgd_baseline_model_ft11_v1.pth')#只保存模型权重参数，不保存模型结构
# model.load_state_dict(torch.load(mymodel.pth))
# model.eval()

# torch.save(model, 'sgd_baseline_model_ft_v2.pth')#保存整个model的状态
# model=torch.load(mymodel.pth)#这里已经不需要重构模型结构了，直接load就可以
# model.eval()




