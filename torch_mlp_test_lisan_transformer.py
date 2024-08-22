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

file17z_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z/17_train_val_lisan_A.csv'
file17z_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z/17_train_val_lisan_B.csv'
file17z_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z/17_train_val_lisan_C.csv'
file17z_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z/17_train_val_lisan_D.csv'
data17z_A = pd.read_csv(file17z_path_A)
data17z_B = pd.read_csv(file17z_path_B)
data17z_C = pd.read_csv(file17z_path_C)
data17z_D = pd.read_csv(file17z_path_D)

file17z_path_yingli_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z/17_train_val_lisan_yingli_A.csv'
file17z_path_yingli_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z/17_train_val_lisan_yingli_B.csv'
file17z_path_yingli_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z/17_train_val_lisan_yingli_C.csv'
file17z_path_yingli_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z/17_train_val_lisan_yingli_D.csv'
data17z_yingli_A = pd.read_csv(file17z_path_yingli_A)
data17z_yingli_B = pd.read_csv(file17z_path_yingli_B)
data17z_yingli_C = pd.read_csv(file17z_path_yingli_C)
data17z_yingli_D = pd.read_csv(file17z_path_yingli_D)

file17z_loc_path_A = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z_loc/17_train_val_lisan_A.csv'
file17z_loc_path_B = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z_loc/17_train_val_lisan_B.csv'
file17z_loc_path_C = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z_loc/17_train_val_lisan_C.csv'
file17z_loc_path_D = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z_loc/17_train_val_lisan_D.csv'
data17z_loc_A = pd.read_csv(file17z_loc_path_A)
data17z_loc_B = pd.read_csv(file17z_loc_path_B)
data17z_loc_C = pd.read_csv(file17z_loc_path_C)
data17z_loc_D = pd.read_csv(file17z_loc_path_D)

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
# dataframes_all.append(data12_A)
# dataframes_all.append(data12_B)
# dataframes_all.append(data12_C)
# dataframes_all.append(data12_D)
# dataframes_all.append(data13_A)
# dataframes_all.append(data13_B)
# dataframes_all.append(data13_C)
# dataframes_all.append(data13_D)
# dataframes_all.append(data14_A)
# dataframes_all.append(data14_B)
# dataframes_all.append(data14_C)
# dataframes_all.append(data14_D)
# dataframes_all.append(data15_A)
# dataframes_all.append(data15_B)
# dataframes_all.append(data15_C)
# dataframes_all.append(data15_D)
# dataframes_all.append(data16_A)
# dataframes_all.append(data16_B)
# dataframes_all.append(data16_C)
# dataframes_all.append(data16_D)
# dataframes_all.append(data17_A)
# dataframes_all.append(data17_B)
# dataframes_all.append(data17_C)
# dataframes_all.append(data17_D)
dataframes_all.append(data17z_A)
dataframes_all.append(data17z_B)
dataframes_all.append(data17z_C)
dataframes_all.append(data17z_D)
# dataframes_all.append(data17z_yingli_A)
# dataframes_all.append(data17z_yingli_B)
# dataframes_all.append(data17z_yingli_C)
# dataframes_all.append(data17z_yingli_D)
# dataframes_all.append(data17z_loc_A)
# dataframes_all.append(data17z_loc_B)
# dataframes_all.append(data17z_loc_C)
# dataframes_all.append(data17z_loc_D)

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
dataframes.append(data17z_A)
dataframes.append(data17z_B)
dataframes.append(data17z_C)
dataframes.append(data17z_D)
# dataframes.append(data17z_yingli_A)
# dataframes.append(data17z_yingli_B)
# dataframes.append(data17z_yingli_C)
# dataframes.append(data17z_yingli_D)
# dataframes.append(data17z_loc_A)
# dataframes.append(data17z_loc_B)
# dataframes.append(data17z_loc_C)
# dataframes.append(data17z_loc_D)

# dataframes.append(data17_7524_A)
# dataframes.append(data17_7524_B)
# dataframes.append(data17_7524_C)
# dataframes.append(data17_7524_D)

# dataframes.append(data13_6011_A)
# dataframes.append(data13_6011_B)
# dataframes.append(data13_6011_C)
# dataframes.append(data13_6011_D)

# # Concatenate all dataframes into one
combined_df_all = pd.concat(dataframes_all, ignore_index=True)
combined_df = pd.concat(dataframes, ignore_index=True)
print(combined_df_all.shape, combined_df.shape)


# Standardize the features
scaler = StandardScaler()

# get the training datas
# Define the input features and the target variable
# X = data[['ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
# X_all = combined_df_all[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
X_all_cond = combined_df_all[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION']].to_numpy()
# X_all_UXY = combined_df_all[['UX', 'UY']].to_numpy().astype(np.int_)
# X_all_UXY = combined_df_all[['UX', 'UY']].to_numpy()
# X_all_ROTXY = combined_df_all[['ROTX', 'ROTY']].to_numpy()*57.3
# X_all_ROTXY = np.around(X_all_ROTXY, 2) #使用around()函数保留小数位数
# X_all = np.concatenate((X_all_cond, X_all_UXY), axis=1)
# # X_all = np.concatenate((X_all_cond, X_all_UXY, X_all_ROTXY), axis=1)

# X_all_UXYZ = combined_df_all[['UX_1', 'UY_1', 'UZ_1', 'UX_2', 'UY_2', 'UZ_2']].to_numpy().astype(np.int_)
# X_all_UXYZ = combined_df_all[['UX_1', 'UY_1', 'UZ_1']].to_numpy().astype(np.int_)
# X_all_UXYZ = combined_df_all[['UX_1', 'UY_1', 'UZ_1', 'UX_3', 'UY_3', 'UZ_3']].to_numpy().astype(np.int_)
# X_all_UXYZ = combined_df_all[['UX_1', 'UY_1', 'UZ_1', 'UX_3', 'UY_3', 'UZ_3']].to_numpy()
# X_all_UXYZ = combined_df_all[['UX_2', 'UY_2', 'UZ_2', 'UX_3', 'UY_3', 'UZ_3']].to_numpy().astype(np.int_)
# X_all_UXYZ = combined_df_all[['UX_2', 'UY_2', 'UZ_2', 'UX_3', 'UY_3', 'UZ_3']].to_numpy()

X_all_UXYZ = combined_df_all[['UX', 'UY', 'UZ']].to_numpy().astype(np.int_)
X_all_UXYZ = (X_all_UXYZ //10 ) * 10
# X_all_UXYZ = combined_df_all[['UX', 'UY', 'UZ', 'A19', 'B19', 'C19', 'D19']].to_numpy().astype(np.int_)
# X_all_UXYZ = combined_df_all[['UX', 'UY', 'UZ']].to_numpy()
# X_all_ROTXYZ = combined_df_all[['ROTX', 'ROTY', 'ROTZ']].to_numpy()*57.3
# X_all_ROTXYZ = np.around(X_all_ROTXYZ, 2) #使用around()函数保留小数位数
# X_all = np.concatenate((X_all_cond, X_all_UXYZ, X_all_ROTXYZ), axis=1)
X_all = np.concatenate((X_all_cond, X_all_UXYZ), axis=1)

# print('X_all.shape=========', X_all.shape)
y_all = combined_df_all[['DAMAGE_LOCATION', 'LABEL']].to_numpy()
# y_all_loc = y_all[:,0]
# y_all_loc[y_all_loc>0]=1
# y_all[:,0] = y_all_loc

# X_all_scaled = scaler.fit_transform(X_all)
X_all_scaled = StandardScaler().fit(X_all)

# X = combined_df[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
# X_cond = combined_df[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION']].to_numpy()
# X_UXY = combined_df[['UX', 'UY']].to_numpy().astype(np.int_)
# # X_UXY = combined_df[['UX', 'UY']].to_numpy()
# X_ROTXY = combined_df[['ROTX', 'ROTY']].to_numpy()*57.3
# X_ROTXY = np.around(X_ROTXY, 2)
# X = np.concatenate((X_cond, X_UXY), axis=1)
# # X = np.concatenate((X_cond, X_UXY, X_ROTXY), axis=1)

# print('use the float UXUYZ !!!!!!!!')
# print('use the int UXUYZ and round 2 ROTXYZ !!!!!!!!')
# print('use the int UXUYZ and original ROTXYZ !!!!!!!!')
# print('use the float UXUYZ and original ROTXYZ !!!!!!!!')
X_cond = combined_df[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION']].to_numpy()
# print('=======UX_1+UX_2 int==========')
# print('=======UX_1+UX_2==========')
# print('=======UX_1+UX_3 int==========')
# print('=======UX_1+UX_3==========')
# print('=======UX_2+UX_3 int==========')
# print('=======UX_2+UX_3==========')
# X_UXYZ = combined_df[['UX_1', 'UY_1', 'UZ_1', 'UX_2', 'UY_2', 'UZ_2']].to_numpy().astype(np.int_)
# X_UXYZ = combined_df[['UX_1', 'UY_1', 'UZ_1']].to_numpy().astype(np.int_)
# X_UXYZ = combined_df[['UX_1', 'UY_1', 'UZ_1', 'UX_3', 'UY_3', 'UZ_3']].to_numpy().astype(np.int_)
# X_UXYZ = combined_df[['UX_1', 'UY_1', 'UZ_1', 'UX_3', 'UY_3', 'UZ_3']].to_numpy()
# X_UXYZ = combined_df[['UX_2', 'UY_2', 'UZ_2', 'UX_3', 'UY_3', 'UZ_3']].to_numpy().astype(np.int_)
# X_UXYZ = combined_df[['UX_2', 'UY_2', 'UZ_2', 'UX_3', 'UY_3', 'UZ_3']].to_numpy()

X_UXYZ = combined_df[['UX', 'UY', 'UZ']].to_numpy().astype(np.int_)
X_UXYZ = (X_UXYZ //10 ) * 10
# X_UXYZ = combined_df[['UX', 'UY', 'UZ', 'A19', 'B19', 'C19', 'D19']].to_numpy().astype(np.int_)
# X_UXYZ = combined_df[['UX', 'UY', 'UZ']].to_numpy()
# X_ROTXYZ = combined_df[['ROTX', 'ROTY', 'ROTZ']].to_numpy()*57.3
# X_ROTXYZ = np.around(X_ROTXYZ, 2)
# X = np.concatenate((X_cond, X_UXYZ, X_ROTXYZ), axis=1)
X = np.concatenate((X_cond, X_UXYZ), axis=1)

y = combined_df[['DAMAGE_LOCATION', 'LABEL']].to_numpy()
# y_loc = y[:,0]
# y_loc[y_loc>0]=1
# y[:,0] = y_loc

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
# txtsave = '/mnt/d/jinyfeng/datas/jinyfeng(2)/y_test.txt'
# with open(txtsave,'w') as f:
#     f.writelines(str(y_test))
print(X_train.shape, y_train.shape)
print(X_test.shape, len(X_test), type(y_test), y_test.shape, len(y_test))

X_train_scaled = X_all_scaled.transform(X_train)
X_test_scaled = X_all_scaled.transform(X_test)
# X_train_scaled = X_train_scaled.to_numpy()
# X_test_scaled = X_test_scaled.to_numpy()

train_dataset = SelfDataset(X_train_scaled, y_train)
test_dataset = SelfDataset(X_test_scaled, y_test)
batch_size=64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        self.layers = nn.Sequential(
            nn.Linear(8, 80),
            activate_layer,
            nn.Linear(80, 40),
            activate_layer,
        )
        self.dam_loc = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=40, out_features=19)
        )
        self.ori_loc = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Linear(in_features=40, out_features=4)
        )

        self.fc1 = torch.nn.Linear(8,80)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(80,40)  # 第二个隐含层
        self.fc3 = torch.nn.Linear(40,19)   # 输出层
        self.fc4 = torch.nn.Linear(40,4)   # 输出层
        
    def forward(self, din):
        # 前向传播， 输入值：din, 返回值 dout
        # print('din.shape1111111=====', din.shape)
        # din = din.view(-1, 8)       # 将一个多行的Tensor,拼接成一行
        # print('din.shape222222=====', din.shape)
        dout = self.layers(din)
        dout1 = self.dam_loc(dout)
        dout2 = self.ori_loc(dout)
        # dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        # dout = F.relu(self.fc2(dout))
        # dout1 = self.fc3(dout)
        # dout2 = self.fc4(dout)

        # dout1 = F.softmax(self.fc3(dout), dim=1)  # 输出层使用 softmax 激活函数
        # dout2 = F.softmax(self.fc4(dout), dim=1)  # 输出层使用 softmax 激活函数
        return dout1, dout2

def init_weights(model_param):
    if type(model_param) == nn.Linear or type(model_param) == nn.Conv2d:
        # nn.init.xavier_uniform_(model_param.weight)
        nn.init.kaiming_normal_(model_param.weight)

# model = MLP()
# model = TransformerClassifier(6, 19, 4)
model = TransformerClassifier(7, 19, 4)
# model = TransformerClassifier(8, 19, 4)
# model = TransformerClassifier(10, 19, 4)
# model = TransformerClassifier(11, 19, 4)

# model.load_state_dict(torch.load('torch_model_transformer_7020_12-17XY_int.pth'))
model.load_state_dict(torch.load('torch_model_transformer_7020_17XYZ_int_v1.pth'))
# model.load_state_dict(torch.load('torch_model_transformer_XYZ_loc_v0.pth'))
# model.load_state_dict(torch.load('torch_model_transformer_XYZ_loc_ABCD19_v0.pth'))
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device=====', device)
if torch.cuda.is_available():
    # model = model.cuda()
    model.to(device)


test_result = '/mnt/d/jinyfeng/datas/jinyfeng(2)/17-z/17_val_lisan.csv'
df_save = pd.DataFrame()
# X_test = np.array(X_test)
df_save[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'UZ']] = X_test
# df_save[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'UZ', 'A19', 'B19', 'C19', 'D19']] = X_test
df_save[['DAMAGE_LOCATION', 'LABEL']] = y_test

y_test_pred1 = []
y_test_pred2 = []
correct_loc, correct_or, correct, total = 0, 0, 0, 0
equal_elements_test = 0
model.eval()
with torch.no_grad():  # 训练集中不需要反向传播
    for data, labels in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()
        output1, output2 = model(data)
        # print(output1.shape, output2.shape)
        # print(torch.max(output1.data, 1))
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        # print(predicted1.shape, type(predicted1))
        # print(labels, labels.shape, labels.size(0))
        y_test_pred1.extend(predicted1.cpu())
        y_test_pred2.extend(predicted2.cpu())

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

# print(np.array(y_test_pred1).shape, np.array(y_test_pred2).shape)
df_save['DAMAGE_LOCATION_pred'] = np.array(y_test_pred1)
df_save['LABEL_pred'] = np.array(y_test_pred2)
df_save.to_csv(test_result, mode = 'w', index =False)   #保存到csv,  mode=a，以追加模式写入,header表示列名，默认为true,index表示行名，默认为true，再次写入不需要行名

# temp = '教程是：%(name)s, 价格是：%(price).2f, 网址是：%(url)s'
# course = {'name':'Python教程', 'price': 9.9, 'url': 'http://c.biancheng.net/python/'}
# # 使用字典为字符串模板中的key传入值
# print(temp % course)
# course = {'name':'C++教程', 'price':15.6, 'url': 'http://c.biancheng.net/cplus/'}
# # 使用字典为字符串模板中的key传入值
# print(temp % course)  # 这里的%本质也是分隔符


