import pandas as pd
import numpy as np
from joblib import dump

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


dataframes.append(data7_A)
dataframes.append(data7_B)
dataframes.append(data7_C)
dataframes.append(data7_D)
# dataframes.append(data8_A)
# dataframes.append(data8_B)
# dataframes.append(data8_C)
# dataframes.append(data8_D)
# dataframes.append(data9_A)
# dataframes.append(data9_B)
# dataframes.append(data9_C)
# dataframes.append(data9_D)
# dataframes.append(data10_A)
# dataframes.append(data10_B)
# dataframes.append(data10_C)
# dataframes.append(data10_D)
dataframes.append(data11_A)
dataframes.append(data11_B)
dataframes.append(data11_C)
dataframes.append(data11_D)
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

# # Concatenate all dataframes into one
combined_df_all = pd.concat(dataframes_all, ignore_index=True)
combined_df = pd.concat(dataframes, ignore_index=True)
print(combined_df_all.shape, combined_df.shape)


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Standardize the features
scaler = StandardScaler()

# get the training datas
# Define the input features and the target variable
# X = data[['ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
X_all = combined_df_all[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
y_all = combined_df_all[['DAMAGE_LOCATION', 'LABEL']]
# X_all = X_all.to_numpy()
# y_all = y_all.to_numpy()
# X_all_scaled = StandardScaler().fit(X_all)

X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, shuffle=True)
print(X_all_train.shape, y_all_train.shape)
print(X_all_test.shape, len(X_all_test), type(y_all_test), y_all_test.shape, len(y_all_test))
X_all_scaled = StandardScaler().fit(X_all_train)

X_all_train_scaled = X_all_scaled.transform(X_all_train)
X_all_test_scaled = X_all_scaled.transform(X_all_test)
print(type(X_all_train_scaled), X_all_train_scaled.shape, type(X_all_test_scaled), X_all_test_scaled.shape)
y_all_train = np.array(y_all_train)
y_all_test = np.array(y_all_test)

X = combined_df[['SECTION_NUMBER', 'ANGLE', 'WEIGHT', 'WEIGHT_LOCATION', 'UX', 'UY', 'ROTX', 'ROTY']]
y = combined_df[['DAMAGE_LOCATION', 'LABEL']]
# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42, shuffle=True)
# txtsave = '/mnt/d/jinyfeng/datas/jinyfeng(2)/y_test.txt'
# with open(txtsave,'w') as f:
#     f.writelines(str(y_test))
print(X_train.shape, y_train.shape)
print(X_test.shape, len(X_test), type(y_test), y_test.shape, len(y_test))

X_train_scaled = X_all_scaled.transform(X_train)
X_test_scaled = X_all_scaled.transform(X_test)
print(type(X_train_scaled), X_train_scaled.shape, type(X_test_scaled), X_test_scaled.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train_scaled = np.concatenate((X_train_scaled, X_all_train_scaled), axis=0)
y_train = np.concatenate((y_train, y_all_train), axis=0)
print(type(X_train_scaled), X_train_scaled.shape, type(y_train), y_train.shape)
# # Train the MLP model
# Initialize the MLPClassifier with a random state for reproducibility
# mlp = MLPClassifier(hidden_layer_sizes=(256), max_iter=1000, random_state=42)
# mlp = MLPClassifier(hidden_layer_sizes=(128,64), activation='logistic', max_iter=1000, solver='sgd', random_state=3)
# mlp =  MLPClassifier(hidden_layer_sizes=(100,50), alpha=1e-5, max_iter=2000)
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(80, 40), alpha=1e-5, max_iter=40000, max_fun=80000)
# mlp = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, activation='logistic', solver='adam', random_state=42)
# solver='lbfgs', 'adam', 'sgd', alpha=1e-4, 

clf = MultiOutputClassifier(mlp).fit(X_train_scaled, y_train)
# mlp.fit(X_train_scaled, y_train)

dump(clf, "mlp_train_model_lisan_12-17_ft7_11_iter4w_v1.pkl")

# Predict the damage location for both training and testing sets
y_train_pred = clf.predict(X_train_scaled)
y_test_pred = clf.predict(X_test_scaled)
print(len(y_test), y_test.shape, y_test_pred.shape)
# print(X_test)
# print(y_test, y_test.shape)
# print(y_test_pred, y_test_pred.shape)

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

# equal_elements_train = sum([1 for i, j in zip(y_train, y_train_pred) if i.tolist() == j.tolist()])
# train_acc = equal_elements_train / len(y_train)
# train_acc1 = equal_elements_train / y_train.shape[0]
# equal_elements_test = sum([1 for i, j in zip(y_test, y_test_pred) if i.tolist() == j.tolist()])
# test_acc = equal_elements_test / len(y_test)
# test_acc1 = equal_elements_test / y_test.shape[0]
# print('train_acc, test_acc=========', train_acc, test_acc)
# print('train_acc, test_acc=========', train_acc1, test_acc1)

y_all_train_pred = clf.predict(X_all_train_scaled)
y_all_test_pred = clf.predict(X_all_test_scaled)
pretrain_train_accuracy_0 = accuracy_score(y_all_train[:,0], y_all_train_pred[:,0])
pretrain_train_accuracy_1 = accuracy_score(y_all_train[:,1], y_all_train_pred[:,1])
pretrain_test_accuracy_0 = accuracy_score(y_all_test[:,0], y_all_test_pred[:,0])
pretrain_test_accuracy_1 = accuracy_score(y_all_test[:,1], y_all_test_pred[:,1])
print('pretrain_train_accuracy_0, pretrain_train_accuracy_1=========', pretrain_train_accuracy_0, pretrain_train_accuracy_1)
print('pretrain_test_accuracy_0, pretrain_test_accuracy_1=========', pretrain_test_accuracy_0, pretrain_test_accuracy_1)

pretrain_train_score = clf.score(X_all_train_scaled, y_all_train)
pretrain_test_score = clf.score(X_all_test_scaled, y_all_test)
print('pretrain_train_score, pretrain_test_score=========', pretrain_train_score, pretrain_test_score)


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






# # SVM
# # Define the features and the target variable for the new data

# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# # Train an SVM classifier on the new training data
# svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
# svm_classifier.fit(X_train_scaled, y_train)

# # Predict the DAMAGE_LOCATION for the new test set
# svm_predictions_train = svm_classifier.predict(X_train_scaled)
# svm_predictions_test = svm_classifier.predict(X_test_scaled)

# # Calculate the accuracy of the SVM classifier on the new test set
# svm_accuracy_train = accuracy_score(y_train, svm_predictions_train)
# svm_accuracy_test = accuracy_score(y_test, svm_predictions_test)
# print('svm_accuracy_train, svm_accuracy_test==========', svm_accuracy_train, svm_accuracy_test)





# # #随机森林，决策树

# # Initialize the Random Forest classifier
# rf_classifier = RandomForestClassifier(random_state=42)
# # Train the Random Forest classifier
# rf_classifier.fit(X_train_scaled, y_train)
# # Make predictions using the Random Forest classifier
# rf_pred_train = rf_classifier.predict(X_train_scaled)
# rf_pred_test = rf_classifier.predict(X_test_scaled)

# # Initialize the Decision Tree classifier
# dt_classifier = DecisionTreeClassifier(random_state=42)
# # Train the Decision Tree classifier
# dt_classifier.fit(X_train_scaled, y_train)
# # Make predictions using the Decision Tree classifier
# dt_pred_train = dt_classifier.predict(X_train_scaled)
# dt_pred_test = dt_classifier.predict(X_test_scaled)

# # Calculate the accuracy for both classifiers
# rf_accuracy_train = accuracy_score(y_train, rf_pred_train)
# rf_accuracy_test = accuracy_score(y_test, rf_pred_test)

# dt_accuracy_train = accuracy_score(y_train, dt_pred_train)
# dt_accuracy_test = accuracy_score(y_test, dt_pred_test)

# print('rf_accuracy_train, rf_accuracy_test=========', rf_accuracy_train, rf_accuracy_test)
# print('dt_accuracy_train, dt_accuracy_test=========', dt_accuracy_train, dt_accuracy_test)




