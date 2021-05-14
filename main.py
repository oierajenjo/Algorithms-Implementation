import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

from PCA.reductionPCA import reductionPCA, set_train_validationData, read_csv_data, preprocess_data, plotting, \
    standardise, pca_df
from RBFN.classifierRBFN import train_data, measure_accuracy, get_XY, amount_centroids
from Resources.retrieveData import download_file_from_google_drive

file_all = os.getcwd() + '/data/allData.csv'
file_noisy = os.getcwd() + '/data/allNoisyData.csv'
file_validation = os.getcwd() + '/data/validationData.csv'
file_train = os.getcwd() + '/data/trainData.csv'


"""
CHECK IF FILES EXIST
If not execute download_file_from_google_drive
"""
if not os.path.exists(file_all):
    download_file_from_google_drive('allData.csv')

if not os.path.exists(file_noisy):
    download_file_from_google_drive('allNoisyData.csv')


"""
SEPARATE DATA
Data samples are separated in train, validation and test data
"""
if not os.path.exists(file_validation) or not os.path.exists(file_train):
    # Importing Dataset
    df = pd.read_csv(file_all)
    df.head()
    df = df.loc[(df['HeatLoad'] == 10000) & ((df['T_set'] == 0) | (df['T_set'] == 12))
                & ((df['Cpr_Scale'] >= 0.9) | (df['Cpr_Scale'] == 0.3))]
    # Preprocessing
    X, Y = preprocess_data(df)

    X_train, Y_train, X_validation, Y_validation = set_train_validationData(X, Y)
else:
    X_train, Y_train = read_csv_data(file_train)
    X_validation, Y_validation = read_csv_data(file_validation)

X_test, Y_test = read_csv_data(file_noisy)

amount_pcs = len(X_train.columns)
print(amount_pcs)


"""
STANDARDISATION
Standardisation process were data is standardised to mean 0 and var 1
"""
X_train, X_validation, X_test = standardise(X_train, X_validation, X_test)


"""
PCA
If don't want to implement PCA comment this part
"""
accuracy = 0.995
X_train, X_validation, X_test, explained_variance, amount_pcs = reductionPCA(X_train, X_validation, X_test, accuracy)


"""
PCA Matrix
Generate 'PCA' matrix
"""
pca_train = pca_df(X_train, Y_train, amount_pcs)
pca_validation = pca_df(X_validation, Y_validation, amount_pcs)
pca_test = pca_df(X_test, Y_test, amount_pcs)
print("Final PCA X")
print(pca_train)

# Plotting PCA training data
fig, ax = plotting(pca_train)
ax.set_title('3 component PCA Train', fontsize=20)
ax.legend(['Non Faulty', 'Faulty'])
fig.show()
fig.savefig('results/3d(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')


"""
RBFN
Instead of validation use test data
If don't want to implement RBFN comment this part
"""
# Accuracy per amount of centroid
centroids_max = 20
step = 1
scores, c_all = amount_centroids(pca_train, pca_validation, amount_pcs, centroids_max, step)

# Plotting accuracy per Amount of Centroids
plt.plot(c_all, scores)
plt.title('Accuracy per Amount of Centroids')
plt.xlabel('Amount of Centroids')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
fig.savefig('results/Accuracy-Centroids(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')

"""
RBFN
Accuracy with noisy data
"""
X_pca_train, Y_pca_train = get_XY(pca_train, amount_pcs)
centroids, W, sigma = train_data(X_pca_train, Y_pca_train, n_centroids=c_all[-1])
X_pca_test, Y_pca_test = get_XY(pca_test, amount_pcs)
score = measure_accuracy(X_pca_test, Y_pca_test, centroids, sigma, W)

fig, ax = plotting(pca_test)
ax.set_title('3 component PCA Test', fontsize=20)
ax.legend(['Non Faulty', 'Faulty'])
fig.show()
fig.savefig('results/3d(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')
