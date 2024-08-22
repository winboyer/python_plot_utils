import pandas as pd
import numpy as np
from joblib import dump
from sklearn.datasets import make_moons

from kan import *
from kan import KAN, create_dataset
import torch

# train_input, train_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)
# print(train_input.dtype, train_label.dtype)
# print(train_input.shape, train_label.shape)
# print(train_input, train_label)
# dataset = {}
# dataset['train_input'] = torch.from_numpy(train_input)
# dataset['train_label'] = torch.from_numpy(train_label[:,None])
# print(dataset['train_input'].dtype, dataset['train_input'].shape)
# print(dataset['train_label'].dtype, dataset['train_label'].shape)

# f = lambda x: torch.exp((torch.sin(torch.pi*(x[:,[0]]**2+x[:,[1]]**2))+torch.sin(torch.pi*(x[:,[2]]**2+x[:,[3]]**2)))/2)
# dataset = create_dataset(f, n_var=4, train_num=3000)
# # print(dataset['train_input'],dataset['test_input'],dataset['train_label'],dataset['test_label'])
# print(dataset['train_input'].shape, dataset['train_input'].dtype,
#     dataset['test_input'].shape, dataset['test_input'].dtype,
#     dataset['train_label'].shape, dataset['train_label'].dtype,
#     dataset['test_label'].shape, dataset['test_label'].dtype)






# Load the uploaded Excel file
# file1_path = '/Users/jinyfeng/Downloads/jinyfeng(2)/train_val_seq_v1.xls'
file1_path = 'D:/jinyfeng/datas/jinyfeng(2)/train_val_seq_v1.xls'
# file2_path = '/Users/jinyfeng/Downloads/jinyfeng(2)/working condition-17-225_seq.xls'
# file3_path = '/Users/jinyfeng/Downloads/jinyfeng(2)/working condition-17-270_seq.xls'
# file4_path = '/Users/jinyfeng/Downloads/jinyfeng(2)/working condition-17-315_seq.xls'

data = pd.read_excel(file1_path)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# dataframes = [data]
# df2 = pd.read_excel(file2_path)
# dataframes.append(df2)
# df3 = pd.read_excel(file3_path)
# dataframes.append(df3)
# df4 = pd.read_excel(file4_path)
# dataframes.append(df4)

# # Concatenate all dataframes into one
# combined_df = pd.concat(dataframes, ignore_index=True)
# # print(combined_df.shape)

# Display the distribution of DAMAGE_LOCATION values
# damage_location_counts = combined_df['DAMAGE_LOCATION'].value_counts()

# MLP
# Define the input features and the target variable
X = data[['WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY',
            'UX_45', 'UY_45', 'ROTX_45','ROTY_45',
            'UX_90', 'UY_90', 'ROTX_90','ROTY_90',
            'UX_135', 'UY_135', 'ROTX_135','ROTY_135',
            'UX_180', 'UY_180', 'ROTX_180','ROTY_180',
            'UX_225', 'UY_225', 'ROTX_225','ROTY_225',
            'UX_270', 'UY_270', 'ROTX_270','ROTY_270',
            'UX_315', 'UY_315', 'ROTX_315','ROTY_315']]

# X = data[['WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY',
#             'WEIGHT_45', 'WEIGHT_LOCATION_45', 'UX_45', 'UY_45', 'ROTX_45','ROTY_45',
#             'WEIGHT_90', 'WEIGHT_LOCATION_90', 'UX_90', 'UY_90', 'ROTX_90','ROTY_90',
#             'WEIGHT_135', 'WEIGHT_LOCATION_135', 'UX_135', 'UY_135', 'ROTX_135','ROTY_135',
#             'WEIGHT_180', 'WEIGHT_LOCATION_180', 'UX_180', 'UY_180', 'ROTX_180','ROTY_180',
#             'WEIGHT_225', 'WEIGHT_LOCATION_225', 'UX_225', 'UY_225', 'ROTX_225','ROTY_225',
#             'WEIGHT_270', 'WEIGHT_LOCATION_270', 'UX_270', 'UY_270', 'ROTX_270','ROTY_270',
#             'WEIGHT_315', 'WEIGHT_LOCATION_315', 'UX_315', 'UY_315', 'ROTX_315','ROTY_315']]

y = data['DAMAGE_LOCATION']
print('X.shape, y.shape========', X.shape, y.shape)


# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# txtsave = '/Users/jinyfeng/Downloads/jinyfeng(2)/y_test.txt'
# with open(txtsave,'w') as f:
#     f.writelines(str(y_test))

# print(X_test.shape, len(X_test), type(y_test), y_test.shape, len(y_test))

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
# model = KAN(width=[4,2,1,1], grid=3, k=3, seed=0)
# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
# model = KAN(width=[4, 9, 1], grid=3, k=3, seed=0)
model = KAN(width=[34, 39, 1], grid=3, k=3, seed=0)

dataset = {}
dataset['train_input'] = torch.from_numpy(np.array(X_train_scaled))
dataset['test_input'] = torch.from_numpy(np.array(X_test_scaled))
dataset['train_label'] = torch.from_numpy(np.array(y_train))
dataset['test_label'] = torch.from_numpy(np.array(y_test))
print(dataset['train_input'].shape, dataset['train_input'].dtype)
print(dataset['train_label'].shape, dataset['train_label'].dtype)
# train the model
results = model.train(dataset, opt="LBFGS", steps=20, lamb=0.001)
# results = model.train(dataset, opt="LBFGS", steps=20, lamb=0.001, lamb_entropy=2.)
def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())
def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())
# results = model.train(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss())
print('results=======', results)


grids = np.array([5,10,20,50,100])

train_losses = []
test_losses = []
steps = 50
k = 3

for i in range(grids.shape[0]):
    if i == 0:
        model = KAN(width=[2,1,1], grid=grids[i], k=k)
    if i != 0:
        model = KAN(width=[2,1,1], grid=grids[i], k=k).initialize_from_another_model(model, dataset['train_input'])
    results = model.train(dataset, opt="LBFGS", steps=steps, stop_grid_update_step=30)
    train_losses += results['train_loss']
    test_losses += results['test_loss']


grids = [3,5,10,20,50]
#grids = [5]

train_rmse = []
test_rmse = []

for i in range(len(grids)):
    model = KAN(width=[4,9,1], grid=grids[i], k=3, seed=0).initialize_from_another_model(model, dataset['train_input'])
    results = model.train(dataset, opt="LBFGS", steps=50, stop_grid_update_step=30);
    train_rmse.append(results['train_loss'][-1].item())
    test_rmse.append(results['test_loss'][-1].item())










# Train the model
# mlp.fit(X_train, y_train)
mlp.fit(X_train_scaled, y_train)

# Predict the damage location for both training and testing sets
y_train_pred = mlp.predict(X_train_scaled)
y_test_pred = mlp.predict(X_test_scaled)


# Calculate the accuracy for training and testing sets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(len(y_test), y_test.shape, y_test_pred.shape)
print(y_test_pred)
print(train_accuracy, test_accuracy)

loss_curve = mlp.loss_curve_
# print(loss_curve)
# Plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(len(loss_curve)), loss_curve)
plt.title('Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


# dump(mlp, "mlp_train_model.pkl")


# savepath = '/Users/jinyfeng/Downloads/jinyfeng(2)/train_val_seq_pred.csv'
# df = pd.DataFrame()
# X_test = np.array(X_test)
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















# SVM
# Define the features and the target variable for the new data

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# # Define the features and the target variable
# features = data[['ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
# target = data['DAMAGE_LOCATION']

# # Split the dataset into training and testing sets (80% training, 20% testing)
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # Standardize the features (mean=0 and variance=1) to improve model performance
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Train an SVM classifier on the new training data
# svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
# svm_classifier.fit(X_train_scaled, y_train)

# # Predict the DAMAGE_LOCATION for the new test set
# svm_predictions_new = svm_classifier.predict(X_test_scaled)

# # Calculate the accuracy of the SVM classifier on the new test set
# svm_accuracy_new = accuracy_score(y_test, svm_predictions_new)
# print(svm_accuracy_new)





# #随机森林，决策树

# # Define the features and the target
# features = data[['ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
# target = data['DAMAGE_LOCATION']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # Initialize the Random Forest classifier
# rf_classifier = RandomForestClassifier(random_state=42)
# # Train the Random Forest classifier
# rf_classifier.fit(X_train, y_train)
# # Make predictions using the Random Forest classifier
# rf_predictions = rf_classifier.predict(X_test)

# # Initialize the Decision Tree classifier
# dt_classifier = DecisionTreeClassifier(random_state=42)
# # Train the Decision Tree classifier
# dt_classifier.fit(X_train, y_train)
# # Make predictions using the Decision Tree classifier
# dt_predictions = dt_classifier.predict(X_test)

# # Calculate the accuracy for both classifiers
# rf_accuracy = accuracy_score(y_test, rf_predictions)
# dt_accuracy = accuracy_score(y_test, dt_predictions)

# print(rf_accuracy, dt_accuracy)



