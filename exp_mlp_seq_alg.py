import pandas as pd
import numpy as np
from joblib import dump

# Load the uploaded Excel file
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
# X = data[['WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY',
#             'UX_45', 'UY_45', 'ROTX_45','ROTY_45',
#             'UX_90', 'UY_90', 'ROTX_90','ROTY_90',
#             'UX_135', 'UY_135', 'ROTX_135','ROTY_135',
#             'UX_180', 'UY_180', 'ROTX_180','ROTY_180',
#             'UX_225', 'UY_225', 'ROTX_225','ROTY_225',
#             'UX_270', 'UY_270', 'ROTX_270','ROTY_270',
#             'UX_315', 'UY_315', 'ROTX_315','ROTY_315']]

X = data[['WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY',
            'WEIGHT_45', 'WEIGHT_LOCATION_45', 'UX_45', 'UY_45', 'ROTX_45','ROTY_45',
            'WEIGHT_90', 'WEIGHT_LOCATION_90', 'UX_90', 'UY_90', 'ROTX_90','ROTY_90',
            'WEIGHT_135', 'WEIGHT_LOCATION_135', 'UX_135', 'UY_135', 'ROTX_135','ROTY_135',
            'WEIGHT_180', 'WEIGHT_LOCATION_180', 'UX_180', 'UY_180', 'ROTX_180','ROTY_180',
            'WEIGHT_225', 'WEIGHT_LOCATION_225', 'UX_225', 'UY_225', 'ROTX_225','ROTY_225',
            'WEIGHT_270', 'WEIGHT_LOCATION_270', 'UX_270', 'UY_270', 'ROTX_270','ROTY_270',
            'WEIGHT_315', 'WEIGHT_LOCATION_315', 'UX_315', 'UY_315', 'ROTX_315','ROTY_315']]

y = data['DAMAGE_LOCATION']
print(X.shape, y.shape)


# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
# txtsave = '/Users/jinyfeng/Downloads/jinyfeng(2)/y_test.txt'
# with open(txtsave,'w') as f:
#     f.writelines(str(y_test))

# print(X_test.shape, len(X_test), type(y_test), y_test.shape, len(y_test))

# Initialize the MLPClassifier with a random state for reproducibility
# mlp = MLPClassifier(hidden_layer_sizes=(256), max_iter=1000, random_state=42)
# mlp = MLPClassifier(hidden_layer_sizes=(128,64), activation='logistic', max_iter=1000, solver='sgd', random_state=3)
mlp =  MLPClassifier(hidden_layer_sizes=(100,50), alpha=1e-5, max_iter=1000)
# mlp = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, activation='logistic', solver='adam', random_state=42)
# solver='lbfgs', 'adam', 'sgd', alpha=1e-4,activation='relu','tanh','logistic', 'identity'

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
