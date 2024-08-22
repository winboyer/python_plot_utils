import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the uploaded Excel file
file1_path = 'D:/jinyfeng/datas/jinyfeng(2)/train_val_seq_v1.xls'
# file2_path = '/Users/jinyfeng/Downloads/jinyfeng(2)/working condition-17-225_seq.xls'
# file3_path = '/Users/jinyfeng/Downloads/jinyfeng(2)/working condition-17-270_seq.xls'
# file4_path = '/Users/jinyfeng/Downloads/jinyfeng(2)/working condition-17-315_seq.xls'

data = pd.read_excel(file1_path)
X = data[['WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY',
            'WEIGHT_45', 'WEIGHT_LOCATION_45', 'UX_45', 'UY_45', 'ROTX_45','ROTY_45',
            'WEIGHT_90', 'WEIGHT_LOCATION_90', 'UX_90', 'UY_90', 'ROTX_90','ROTY_90',
            'WEIGHT_135', 'WEIGHT_LOCATION_135', 'UX_135', 'UY_135', 'ROTX_135','ROTY_135',
            'WEIGHT_180', 'WEIGHT_LOCATION_180', 'UX_180', 'UY_180', 'ROTX_180','ROTY_180',
            'WEIGHT_225', 'WEIGHT_LOCATION_225', 'UX_225', 'UY_225', 'ROTX_225','ROTY_225',
            'WEIGHT_270', 'WEIGHT_LOCATION_270', 'UX_270', 'UY_270', 'ROTX_270','ROTY_270',
            'WEIGHT_315', 'WEIGHT_LOCATION_315', 'UX_315', 'UY_315', 'ROTX_315','ROTY_315']]
y = data['DAMAGE_LOCATION']
# print(X.shape, y.shape)

# Standardize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# 使用reshape方法将数组重塑为（11172，8，6）
reshaped_X = np.array(X).reshape(X.shape[0], 8, 6)
print(reshaped_X.shape)
reshaped_X=np.expand_dims(reshaped_X, 3)
print(reshaped_X.shape)
y=np.array(y)
y=np.expand_dims(y, 1)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(reshaped_X, y, test_size=0.2, shuffle=True, random_state=42)
print(X_test.shape, len(X_test), type(y_test), y_test.shape, len(y_test))


model = Sequential()
input_shape = (8, 6, 1)
# 第一层卷积，32个3x3的卷积核，使用ReLU激活函数
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
# 第二层卷积，64个3x3的卷积核，使用ReLU激活函数
model.add(Conv2D(64, (3, 3), activation='relu'))
# 最大池化层，2x2的池化窗口
model.add(MaxPooling2D(pool_size=(2, 2)))
# # 添加卷积层，128个3x3的卷积核，使用ReLU激活函数
# model.add(Conv2D(128, (3, 3), activation='relu'))
# # 添加最大池化层，2x2的池化窗口
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # 添加卷积层，256个3x3的卷积核，使用ReLU激活函数
# model.add(Conv2D(256, (3, 3), activation='relu'))
# # 添加最大池化层，2x2的池化窗口
# model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten层，将卷积层的输出展平为一维向量
model.add(Flatten())
# 全连接层，128个节点，使用ReLU激活函数
model.add(Dense(128, activation='relu'))
# Dropout层，防止过拟合
model.add(Dropout(0.5))
# 输出层，根据类别数量设定节点数，使用softmax激活函数
model.add(Dense(19, activation='softmax'))
# 编译模型，选择优化器和损失函数
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(type(X_test), X_test.shape, type(y_test), y_test.shape)
# 假设x_train和y_train是训练数据和标签，x_val和y_val是验证数据和标签
# 训练模型，指定训练数据、标签、验证数据、标签、批次大小和迭代次数
# model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), batch_size=32, epochs=10)
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=10)

# 评估模型的准确率
# accuracy = model.evaluate(X_test_scaled, y_test)[1]
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Validation accuracy: {accuracy:.2f}')

# test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
# print(test_acc)