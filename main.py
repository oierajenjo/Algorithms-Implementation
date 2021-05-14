import os
from datetime import datetime
from os import path

import pandas as pd
from matplotlib import pyplot as plt
from PCA.reductionPCA import reductionPCA, set_train_validationData, read_csv_data, preprocess_data, plotting
from RBFN.classifierRBFN import train_data, measure_accuracy, get_XY, amount_centroids
from Resources.retrieveData import download_file_from_google_drive

file_all = os.getcwd() + '\data\\allData.csv'
file_test = os.getcwd() + '\data\\allNoisyData.csv'
file_validation = os.getcwd() + '\data\\validationData.csv'
file_train = os.getcwd() + '\data\\trainData.csv'

if not path.exists(file_all):
    download_file_from_google_drive('allData.csv')
if not path.exists(file_test):
    download_file_from_google_drive('allNoisyData.csv')

# Train and Test data
if not path.exists(file_validation) or not path.exists(file_train):
    # X, Y = read_csv_data(file_all, conditions="HeatLoad=10000")

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

X_test, Y_test = read_csv_data(file_test)

# PCA
accuracy = 0.995
pca_train, pca_test, pca_validation, explained_variance, amount_pcs = reductionPCA(X_train, Y_train, X_validation
                                                                                   , Y_validation, X_test, Y_test
                                                                                   , accuracy)
# RBFN
# Training
k_max = 300
# step = 10
# scores, ks = amount_centroids(pca_train, pca_validation, amount_pcs, k_max, step)

# # Plotting accuracy per Amount of Centroids
# plt.plot(ks, scores)
# plt.title('Accuracy per Amount of Centroids')
# plt.xlabel('Amount of Centroids')
# plt.ylabel('Accuracy')
# plt.grid()
# plt.show()

# Test
X_pca_train, Y_pca_train = get_XY(pca_train, amount_pcs)
X_pca_validation, Y_pca_validation = get_XY(pca_validation, amount_pcs)
centroids, W, sigma = train_data(X_pca_train, Y_pca_train, k_centroids=k_max)
X_pca_test, Y_pca_test = get_XY(pca_test, amount_pcs)
score = measure_accuracy(X_pca_test, Y_pca_test, centroids, sigma, W)
