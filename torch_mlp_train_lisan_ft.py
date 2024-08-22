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

class SelfDataset(Dataset):  # 定义一个自己的数据集
    def __init__(self, data_input, data_label):
        print('data_input.shape=======', data_input.shape)
        # print(y_data.shape)
        self.len = data_input.shape[0]
        self.x_data = torch.from_numpy(data_input.astype(np.float32))
        self.y_data = torch.from_numpy(data_label)
        print(type(self.x_data), type(self.y_data))
        print((self.x_data).dtype, (self.y_data).dtype)
        print(self.x_data.shape, self.y_data.shape)

    def __getitem__(self, index):  # __getitem__为必须需要实现的一个方法 用于从索引取出数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # __getitem__为必须需要实现的一个方法
        return self.len




dataframes_all = []
dataframes = []
dataframes2 = []

# Load the uploaded Excel file
file7_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/7/7-train_val_lisan_A.csv'
file7_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/7/7-train_val_lisan_B.csv'
file7_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/7/7-train_val_lisan_C.csv'
file7_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/7/7-train_val_lisan_D.csv'
data7_A = pd.read_csv(file7_path_A)
data7_B = pd.read_csv(file7_path_B)
data7_C = pd.read_csv(file7_path_C)
data7_D = pd.read_csv(file7_path_D)

file8_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/8/8-train_val_lisan_A.csv'
file8_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/8/8-train_val_lisan_B.csv'
file8_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/8/8-train_val_lisan_C.csv'
file8_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/8/8-train_val_lisan_D.csv'
data8_A = pd.read_csv(file8_path_A)
data8_B = pd.read_csv(file8_path_B)
data8_C = pd.read_csv(file8_path_C)
data8_D = pd.read_csv(file8_path_D)

file9_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/9/9-train_val_lisan_A.csv'
file9_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/9/9-train_val_lisan_B.csv'
file9_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/9/9-train_val_lisan_C.csv'
file9_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/9/9-train_val_lisan_D.csv'
data9_A = pd.read_csv(file9_path_A)
data9_B = pd.read_csv(file9_path_B)
data9_C = pd.read_csv(file9_path_C)
data9_D = pd.read_csv(file9_path_D)

file10_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/10/10-train_val_lisan_A.csv'
file10_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/10/10-train_val_lisan_B.csv'
file10_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/10/10-train_val_lisan_C.csv'
file10_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/10/10-train_val_lisan_D.csv'
data10_A = pd.read_csv(file10_path_A)
data10_B = pd.read_csv(file10_path_B)
data10_C = pd.read_csv(file10_path_C)
data10_D = pd.read_csv(file10_path_D)

file11_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/11/11-train_val_lisan_A.csv'
file11_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/11/11-train_val_lisan_B.csv'
file11_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/11/11-train_val_lisan_C.csv'
file11_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/11/11-train_val_lisan_D.csv'
data11_A = pd.read_csv(file11_path_A)
data11_B = pd.read_csv(file11_path_B)
data11_C = pd.read_csv(file11_path_C)
data11_D = pd.read_csv(file11_path_D)

file12_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/12/12-train_val_lisan_A.csv'
file12_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/12/12-train_val_lisan_B.csv'
file12_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/12/12-train_val_lisan_C.csv'
file12_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/12/12-train_val_lisan_D.csv'
data12_A = pd.read_csv(file12_path_A)
data12_B = pd.read_csv(file12_path_B)
data12_C = pd.read_csv(file12_path_C)
data12_D = pd.read_csv(file12_path_D)

file13_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/13/13-train_val_lisan_A.csv'
file13_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/13/13-train_val_lisan_B.csv'
file13_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/13/13-train_val_lisan_C.csv'
file13_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/13/13-train_val_lisan_D.csv'
data13_A = pd.read_csv(file13_path_A)
data13_B = pd.read_csv(file13_path_B)
data13_C = pd.read_csv(file13_path_C)
data13_D = pd.read_csv(file13_path_D)

file14_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/14/14-train_val_lisan_A.csv'
file14_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/14/14-train_val_lisan_B.csv'
file14_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/14/14-train_val_lisan_C.csv'
file14_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/14/14-train_val_lisan_D.csv'
data14_A = pd.read_csv(file14_path_A)
data14_B = pd.read_csv(file14_path_B)
data14_C = pd.read_csv(file14_path_C)
data14_D = pd.read_csv(file14_path_D)

file15_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/15/15-train_val_lisan_A.csv'
file15_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/15/15-train_val_lisan_B.csv'
file15_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/15/15-train_val_lisan_C.csv'
file15_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/15/15-train_val_lisan_D.csv'
data15_A = pd.read_csv(file15_path_A)
data15_B = pd.read_csv(file15_path_B)
data15_C = pd.read_csv(file15_path_C)
data15_D = pd.read_csv(file15_path_D)

file16_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/16/16-train_val_lisan_A.csv'
file16_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/16/16-train_val_lisan_B.csv'
file16_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/16/16-train_val_lisan_C.csv'
file16_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/16/16-train_val_lisan_D.csv'
data16_A = pd.read_csv(file16_path_A)
data16_B = pd.read_csv(file16_path_B)
data16_C = pd.read_csv(file16_path_C)
data16_D = pd.read_csv(file16_path_D)

file17_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17/17-train_val_lisan_A.csv'
file17_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17/17-train_val_lisan_B.csv'
file17_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17/17-train_val_lisan_C.csv'
file17_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17/17-train_val_lisan_D.csv'
data17_A = pd.read_csv(file17_path_A)
data17_B = pd.read_csv(file17_path_B)
data17_C = pd.read_csv(file17_path_C)
data17_D = pd.read_csv(file17_path_D)

file17_7524_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/7524/17_train_val_lisan_A.csv'
file17_7524_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/7524/17_train_val_lisan_B.csv'
file17_7524_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/7524/17_train_val_lisan_C.csv'
file17_7524_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/7524/17_train_val_lisan_D.csv'
data17_7524_A = pd.read_csv(file17_7524_path_A)
data17_7524_B = pd.read_csv(file17_7524_path_B)
data17_7524_C = pd.read_csv(file17_7524_path_C)
data17_7524_D = pd.read_csv(file17_7524_path_D)

file13_6011_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/6011/13_train_val_lisan_A.csv'
file13_6011_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/6011/13_train_val_lisan_B.csv'
file13_6011_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/6011/13_train_val_lisan_C.csv'
file13_6011_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/6011/13_train_val_lisan_D.csv'
data13_6011_A = pd.read_csv(file13_6011_path_A)
data13_6011_B = pd.read_csv(file13_6011_path_B)
data13_6011_C = pd.read_csv(file13_6011_path_C)
data13_6011_D = pd.read_csv(file13_6011_path_D)

# dataframes_all.append(data7_A)
# dataframes_all.append(data7_B)
# dataframes_all.append(data7_C)
# dataframes_all.append(data7_D)
# dataframes_all.append(data8_A)
# dataframes_all.append(data8_B)
# dataframes_all.append(data8_C)
# dataframes_all.append(data8_D)
# dataframes_all.append(data9_A)
# dataframes_all.append(data9_B)
# dataframes_all.append(data9_C)
# dataframes_all.append(data9_D)
# dataframes_all.append(data10_A)
# dataframes_all.append(data10_B)
# dataframes_all.append(data10_C)
# dataframes_all.append(data10_D)
# dataframes_all.append(data11_A)
# dataframes_all.append(data11_B)
# dataframes_all.append(data11_C)
# dataframes_all.append(data11_D)
dataframes_all.append(data12_A)
dataframes_all.append(data12_B)
dataframes_all.append(data12_C)
dataframes_all.append(data12_D)
dataframes_all.append(data13_A)
dataframes_all.append(data13_B)
dataframes_all.append(data13_C)
dataframes_all.append(data13_D)
dataframes_all.append(data14_A)
dataframes_all.append(data14_B)
dataframes_all.append(data14_C)
dataframes_all.append(data14_D)
dataframes_all.append(data15_A)
dataframes_all.append(data15_B)
dataframes_all.append(data15_C)
dataframes_all.append(data15_D)
dataframes_all.append(data16_A)
dataframes_all.append(data16_B)
dataframes_all.append(data16_C)
dataframes_all.append(data16_D)
dataframes_all.append(data17_A)
dataframes_all.append(data17_B)
dataframes_all.append(data17_C)
dataframes_all.append(data17_D)


# dataframes.append(data7_A)
# dataframes.append(data7_B)
# dataframes.append(data7_C)
# dataframes.append(data7_D)
# dataframes.append(data11_A)
# dataframes.append(data11_B)
# dataframes.append(data11_C)
# dataframes.append(data11_D)
# dataframes.append(data12_A)
# dataframes.append(data12_B)
# dataframes.append(data12_C)
# dataframes.append(data12_D)
# dataframes.append(data13_A)
# dataframes.append(data13_B)
# dataframes.append(data13_C)
# dataframes.append(data13_D)
# dataframes.append(data14_A)
# dataframes.append(data14_B)
# dataframes.append(data14_C)
# dataframes.append(data14_D)
# dataframes.append(data15_A)
# dataframes.append(data15_B)
# dataframes.append(data15_C)
# dataframes.append(data15_D)
# dataframes.append(data16_A)
# dataframes.append(data16_B)
# dataframes.append(data16_C)
# dataframes.append(data16_D)
# dataframes.append(data17_A)
# dataframes.append(data17_B)
# dataframes.append(data17_C)
# dataframes.append(data17_D)

dataframes.append(data17_7524_A)
dataframes.append(data17_7524_B)
dataframes.append(data17_7524_C)
dataframes.append(data17_7524_D)

dataframes2.append(data13_6011_A)
dataframes2.append(data13_6011_B)
dataframes2.append(data13_6011_C)
dataframes2.append(data13_6011_D)

# # Concatenate all dataframes into one
combined_df_all = pd.concat(dataframes_all, ignore_index=True)
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df2 = pd.concat(dataframes2, ignore_index=True)
print('combined_df_all.shape, combined_df.shape============', combined_df_all.shape, combined_df.shape, combined_df2.shape)

# Standardize the features
scaler = StandardScaler()

# get the training datas
# Define the input features and the target variable
# X = data[['ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
# X_all = combined_df_all[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
# y_all = combined_df_all[['DAMAGE_LOCATION', 'LABEL']]
X_all_cond = combined_df_all[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION']].to_numpy()
X_all_UXY = combined_df_all[['UX', 'UY']].to_numpy().astype(np.int_)
X_all = np.concatenate((X_all_cond, X_all_UXY), axis=1)
y_all = combined_df_all[['DAMAGE_LOCATION', 'LABEL']].to_numpy()
# X_all_scaled = StandardScaler().fit(X_all)

X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, shuffle=True)
print('all_train.shape========', X_all_train.shape, y_all_train.shape)
print('all_test.shape==========',X_all_test.shape, len(X_all_test), type(y_all_test), y_all_test.shape, len(y_all_test))
X_all_scaled = StandardScaler().fit(X_all_train)

X_all_train_scaled = X_all_scaled.transform(X_all_train)
X_all_test_scaled = X_all_scaled.transform(X_all_test)
print('X_all_train_scaled data =======', type(X_all_train_scaled), X_all_train_scaled.shape, type(X_all_test_scaled), X_all_test_scaled.shape)
y_all_train = np.array(y_all_train)
y_all_test = np.array(y_all_test)

# X = combined_df[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
# y = combined_df[['DAMAGE_LOCATION', 'LABEL']]
X_cond = combined_df[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION']].to_numpy()
X_UXY = combined_df[['UX', 'UY']].to_numpy().astype(np.int_)
X = np.concatenate((X_cond, X_UXY), axis=1)
y = combined_df[['DAMAGE_LOCATION', 'LABEL']].to_numpy()

# X2 = combined_df2[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
# y2 = combined_df2[['DAMAGE_LOCATION', 'LABEL']]
X2_cond = combined_df2[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION']].to_numpy()
X2_UXY = combined_df2[['UX', 'UY']].to_numpy().astype(np.int_)
X2 = np.concatenate((X2_cond, X2_UXY), axis=1)
y2 = combined_df2[['DAMAGE_LOCATION', 'LABEL']].to_numpy()
# count_0 = len(y[y['DAMAGE_LOCATION']==0])
# print('count_0==========', count_0)
# X_0 = X[y2['DAMAGE_LOCATION']==0]
# y_0 = y[y2['DAMAGE_LOCATION']==0]
# X2_0 = X2[y2['DAMAGE_LOCATION']==0]
# y2_0 = y2[y2['DAMAGE_LOCATION']==0]

# X_not0 = X[y['DAMAGE_LOCATION']!=0]
# y_not0 = y[y['DAMAGE_LOCATION']!=0]
# X2_not0 = X2[y2['DAMAGE_LOCATION']!=0]
# y2_not0 = y2[y2['DAMAGE_LOCATION']!=0]

# X_1_2_concat = np.concatenate((X, X2), axis=0)
# X_all_scaled = StandardScaler().fit(X_1_2_concat)
# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42, shuffle=True)
# txtsave = '/mnt/d/jinyfeng/datas/jinyfeng(2)/y_test.txt'
# with open(txtsave,'w') as f:
#     f.writelines(str(y_test))
print('train.shape==========', X_train.shape, X2_train.shape)
print('test.shape==========', X_test.shape, X2_test.shape)

X_train_scaled = X_all_scaled.transform(X_train)
X_test_scaled = X_all_scaled.transform(X_test)
X2_train_scaled = X_all_scaled.transform(X2_train)
X2_test_scaled = X_all_scaled.transform(X2_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
y2_train = np.array(y2_train)
y2_test = np.array(y2_test)

# X_train_scaled = X_all_scaled.transform(X_0)
# X_test_scaled = X_all_scaled.transform(X_not0)
# X2_train_scaled = X_all_scaled.transform(X2_0)
# X2_test_scaled = X_all_scaled.transform(X2_not0)
# y_train = np.array(y_0)
# y_test = np.array(y_not0)
# y2_train = np.array(y2_0)
# y2_test = np.array(y2_not0)

print('X_train_scaled data========', type(X_train_scaled), X_train_scaled.shape, type(X_test_scaled), X_test_scaled.shape)


# X_train_scaled = np.concatenate((X_train_scaled, X_all_train_scaled), axis=0)
# X_train_scaled = np.concatenate((X_train_scaled, X_all_train_scaled, X2_train_scaled), axis=0)
# X_train_scaled = np.concatenate((X_train_scaled, X2_train_scaled), axis=0)
# X_test_scaled = np.concatenate((X_test_scaled, X2_test_scaled), axis=0)

# y_train = np.concatenate((y_train, y_all_train), axis=0)
# y_train = np.concatenate((y_train, y_all_train, y2_train), axis=0)
# y_train = np.concatenate((y_train, y2_train), axis=0)
# y_test = np.concatenate((y_test, y2_test), axis=0)


# ind = np.arange(X_all_train_scaled.shape[0])
# sub_ind = np.random.choice(ind, X_all_train_scaled.shape[0]//100, replace=False)
# # print(len(sub_ind))
# sub_X_all_train_scaled = X_all_train_scaled[sub_ind]
# sub_y_all_train = y_all_train[sub_ind]
# X_train_scaled = np.concatenate((X_train_scaled, X_all_train_scaled), axis=0)
# # X_train_scaled = np.concatenate((X_train_scaled, sub_X_all_train_scaled), axis=0)
# y_train = np.concatenate((y_train, y_all_train), axis=0)
# # y_train = np.concatenate((y_train, sub_y_all_train), axis=0)
print('X_train_scaled data========', type(X_train_scaled), X_train_scaled.shape, type(y_train), y_train.shape)

# train_dataset = SelfDataset(X_train_scaled, y_train)
# test_dataset = SelfDataset(X_test_scaled, y_test)
print('================6011')
train_dataset = SelfDataset(X2_train_scaled, y2_train)
test_dataset = SelfDataset(X2_test_scaled, y2_test)

test_dataset_pretrain = SelfDataset(X_all_test_scaled, y_all_test)

batch_size=64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loader_pretrain = DataLoader(test_dataset_pretrain, batch_size=batch_size, shuffle=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DomainTransformer(nn.Module):
    def __init__(self, input_dim, hidden_num, num_classes1, num_classes2):
        super(DomainTransformer, self).__init__()
        self.less_hidnum = 2
        self.domain_trans_in = nn.Linear(input_dim-self.less_hidnum, hidden_num)
        self.domain_trans_out = nn.Linear(hidden_num, input_dim-self.less_hidnum)
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
        # for p in self.parameters():
        #     p.requires_grad=False
        for p in self.embedding.parameters():
            p.requires_grad=False
        # for p in self.transformer.parameters():
        #     p.requires_grad=False
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

# model = MLP()
model = TransformerClassifier(6, 19, 4)
# model = TransformerClassifier(8, 19, 4)
# model = DomainTransformer(8, 8, 19, 4)

# torch.save(model.state_dict(), 'sgd_baseline_model_ft_v1.pth')#只保存模型权重参数，不保存模型结构
# model.load_state_dict(torch.load('sgd_baseline_model_v1.pth'))
# float 模型
# model.load_state_dict(torch.load('torch_baseline_model_transformer_v1.pth'))
# int 模型
model.load_state_dict(torch.load('torch_model_transformer_7020_12-17XY_int_v0.pth'))


# model.eval()
# torch.save(model, 'sgd_baseline_model_ft_v2.pth')#保存整个model的状态
# model=torch.load(mymodel.pth)#这里已经不需要重构模型结构了，直接load就可以
# model.eval()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device=====', device)
if torch.cuda.is_available():
    # model = model.cuda()
    model.to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
# m_softmax = nn.LogSoftmax(dim=1)
# criterion = nn.NLLLoss()

# for param_i in model.parameters():
#     print(param_i)
    # i.requires_grad=False

learning_rate=0.001
# optimizer = torch.optim.RMSprop(params = model.parameters(), lr=learning_rate, alpha=0.9)
# optimizer = torch.optim.AdamW(params = model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(params = model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
# optimizer = torch.optim.Adam(params = model.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
# optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
# optimizer = torch.optim.LBFGS(model.parameters(), max_iter=20000, lr=1e-5, tolerance_grad=1e-07,tolerance_change=1e-05)
# optimizer = torch.optim.LBFGS(model.parameters(), max_iter=20000, lr=learning_rate, tolerance_grad=1e-07,tolerance_change=1e-05)

def closure():
    train_losses = []
    model.train()
    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    for data, target in train_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        output1, output2 = model(data)

        # loss1 = criterion(m_softmax(output1), target[:,0])
        # loss2 = criterion(m_softmax(output2), target[:,1])

        loss1 = criterion(output1, target[:,0])  
        loss2 = criterion(output2, target[:,1])

        loss = (loss1+loss2)
        # loss = (loss1+loss2)/2
        loss.backward()
        # print(loss1, loss2, loss)
        # print(loss, type(loss), loss.item(), type(loss.item()))

        train_losses.append(loss.item())
        # print("batch loss: {:.9f}".format(loss.item()))
    return np.mean(train_losses)

# 开始训练
n_epochs=200
for epoch in range(n_epochs):
    
    # train_loss = optimizer.step(closure)
    # print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

    train_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        output1, output2 = model(data)    # 得到预测值
        # print('target.shape==========', target.shape)
        loss1 = criterion(output1, target[:,0])
        loss2 = criterion(output2, target[:,1])
        loss = (loss1+loss2)
        # loss = (loss1+loss2)/2
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    print('Epoch:  {}  \tTraining Loss: {:.6f},{:.6f}'.format(epoch + 1, train_loss, loss))
    
    if (epoch+1)%10==0:
        with torch.no_grad():  # 训练集中不需要反向传播
            correct_loc, correct_or, correct, total = 0, 0, 0, 0
            equal_elements_test = 0
            for data, labels in test_loader:
                if torch.cuda.is_available():
                    data = data.cuda()
                    labels = labels.cuda()
                output1, output2 = model(data)
                # print(output1.shape, output2.shape)
                # print(torch.max(output1.data, 1))
                _, predicted1 = torch.max(output1.data, 1)
                _, predicted2 = torch.max(output2.data, 1)
                # print(labels, labels.shape, labels.size(0))
                total += labels.size(0)
                correct_loc += (predicted1 == labels[:,0]).sum().item()
                correct_or += (predicted2 == labels[:,1]).sum().item()
                correct += ((predicted1 == labels[:,0])&(predicted2 == labels[:,1])).sum().item()
                equal_elements_test += sum([1 for i, j in zip(predicted1 == labels[:,0], predicted2 == labels[:,1]) if((i==j)and(j==1))])
                # print(correct_loc, correct_or, correct, equal_elements_test)
                # score_cmp2=np.mean(np.all(y11 == X11_pred, axis=1))
            print(total, correct_loc, correct_or, correct, equal_elements_test)
            print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
            print('Accuracy of the network on the test images: {:.6f}%,{:.6f}%,{:.6f}%'.format(100*correct_loc/total, 100*correct_or/total, 100*correct/total))
        
            # correct_loc, correct_or, correct, total = 0, 0, 0, 0
            # equal_elements_test = 0
            # for data, labels in test_loader_pretrain:
            #     if torch.cuda.is_available():
            #         data = data.cuda()
            #         labels = labels.cuda()
            #     output1, output2 = model(data)
            #     # print(output1.shape, output2.shape)
            #     # print(torch.max(output1.data, 1))
            #     _, predicted1 = torch.max(output1.data, 1)
            #     _, predicted2 = torch.max(output2.data, 1)
            #     # print(labels, labels.shape, labels.size(0))
            #     total += labels.size(0)
            #     correct_loc += (predicted1 == labels[:,0]).sum().item()
            #     correct_or += (predicted2 == labels[:,1]).sum().item()
            #     correct += ((predicted1 == labels[:,0])&(predicted2 == labels[:,1])).sum().item()
            #     equal_elements_test += sum([1 for i, j in zip(predicted1 == labels[:,0], predicted2 == labels[:,1]) if((i==j)and(j==1))])
            #     # print(correct_loc, correct_or, correct, equal_elements_test)
            #     # score_cmp2=np.mean(np.all(y11 == X11_pred, axis=1))
            
            # print(total, correct_loc, correct_or, correct, equal_elements_test)
            # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
            # print('Accuracy of the network on the test images: {:.6f}%,{:.6f}%,{:.6f}%'.format(100*correct_loc/total, 100*correct_or/total, 100*correct/total))
            
        # temp = '教程是：%(name)s, 价格是：%(price).2f, 网址是：%(url)s'
        # course = {'name':'Python教程', 'price': 9.9, 'url': 'http://c.biancheng.net/python/'}
        # # 使用字典为字符串模板中的key传入值
        # print(temp % course)
        # course = {'name':'C++教程', 'price':15.6, 'url': 'http://c.biancheng.net/cplus/'}
        # # 使用字典为字符串模板中的key传入值
        # print(temp % course)  # 这里的%本质也是分隔符

# torch.save(model.state_dict(), 'sgd_baseline_model_ft11_v1.pth')#只保存模型权重参数，不保存模型结构
# model.load_state_dict(torch.load(mymodel.pth))
# model.eval()

# torch.save(model, 'sgd_baseline_model_ft_v2.pth')#保存整个model的状态
# model=torch.load(mymodel.pth)#这里已经不需要重构模型结构了，直接load就可以
# model.eval()



# mlp = MLPClassifier(hidden_layer_sizes=(256), max_iter=1000, random_state=42)
# mlp = MLPClassifier(hidden_layer_sizes=(128,64), activation='logistic', max_iter=1000, solver='sgd', random_state=3)
# mlp =  MLPClassifier(hidden_layer_sizes=(100,50), alpha=1e-5, max_iter=2000)
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(80, 40), alpha=1e-5, max_iter=40000)
# mlp = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, activation='logistic', solver='adam', random_state=42)
# solver='lbfgs', 'adam', 'sgd', alpha=1e-4, 

# clf = MultiOutputClassifier(mlp).fit(X_train_scaled, y_train)
# mlp.fit(X_train_scaled, y_train)

# joblib.dump(clf, "mlp_train_model_lisan_12-17_iter4w_v3.pkl")

# Predict the damage location for both training and testing sets
y_train_pred = clf.predict(X_train_scaled)
y_test_pred = clf.predict(X_test_scaled)
print(len(y_test), y_test.shape, y_test_pred.shape)
# print(X_test)
# print(y_test, y_test.shape)
# print(y_test_pred, y_test_pred.shape)

y_train = np.array(y_train)
y_test = np.array(y_test)

# Calculate the accuracy for training and testing sets
train_accuracy_0 = accuracy_score(y_train[:,0], y_train_pred[:,0])
train_accuracy_1 = accuracy_score(y_train[:,1], y_train_pred[:,1])
test_accuracy_0 = accuracy_score(y_test[:,0], y_test_pred[:,0])
test_accuracy_1 = accuracy_score(y_test[:,1], y_test_pred[:,1])

print('train_accuracy ========= ', train_accuracy_0, train_accuracy_1)
print('test_accuracy ========= ', test_accuracy_0, test_accuracy_1)

train_score = clf.score(X_train_scaled, y_train)
test_score = clf.score(X_test_scaled, y_test)
print('train_score, test_score=========', train_score, test_score)

equal_elements_train = sum([1 for i, j in zip(y_train, y_train_pred) if i.tolist() == j.tolist()])
train_acc = equal_elements_train / len(y_train)
train_acc1 = equal_elements_train / y_train.shape[0]
equal_elements_test = sum([1 for i, j in zip(y_test, y_test_pred) if i.tolist() == j.tolist()])
test_acc = equal_elements_test / len(y_test)
test_acc1 = equal_elements_test / y_test.shape[0]
print('train_acc, test_acc=========', train_acc, test_acc)
print('train_acc, test_acc=========', train_acc1, test_acc1)



# loss_curve = mlp.loss_curve_
# # print(loss_curve)
# # Plot loss curve
# plt.figure(figsize=(10, 5))
# plt.plot(range(len(loss_curve)), loss_curve)
# plt.title('Loss Curve')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()


savepath = '/mnt/d/jinyfeng/datas/jinyfeng(2)/train_val_lisan12-17_pred.csv'
df = pd.DataFrame()
X_test = np.array(X_test)
df[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']] = X_test
df[['DAMAGE_LOCATION', 'LABEL']] = np.array(y_test)
df[['DAMAGE_LOCATION_pred', 'LABEL_pred']] = np.array(y_test_pred)

# df['ANGLE'] = X_test[:,0]
# df['WEIGHT'] = X_test[:,1]
# df['WEIGHT_LOCATION'] = X_test[:,2]
# df['UX'] = X_test[:,3]
# df['UY'] = X_test[:,4]
# df['ROTX'] = X_test[:,5]
# df['ROTY'] = X_test[:,6]
# df['DAMAGE_LOCATION'] = np.array(y_test)
# df['DAMAGE_LOCATION_PRED'] = y_test_pred
# df.to_csv(savepath, mode = 'w', index =False)   #保存到csv,  mode=a，以追加模式写入,header表示列名，默认为true,index表示行名，默认为true，再次写入不需要行名



