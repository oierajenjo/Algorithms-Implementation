import os
from os import path
import pandas as pd

from PCA.reductionPCA import reductionPCA, set_train_setData, read_csv_data
from RBFN.classifierRBFN import classifierRBFN

file = os.getcwd() + '\data\\allData.csv'
file_test = os.getcwd() + '\data\\testData.csv'
file_train = os.getcwd() + '\data\\trainData.csv'

if not path.exists(file):
    os.system(os.getcwd() + '\Resources\\retrieveData.py')

# Importing Dataset
df = pd.read_csv(file)
df.head()
# print("DATASET")
# print(dataset)
if not path.exists(file_test) or not path.exists(file_train):
    print(1)
    X_train, Y_train, X_test, Y_test = set_train_setData(df)
else:
    print(2)
    X_train, Y_train = read_csv_data(file_train)
    X_test, Y_test = read_csv_data(file_test)

accuracy = 0.9995
pca_train, pca_test, explained_variance, amount_pcs = reductionPCA(X_train, Y_train, X_test, Y_test, accuracy)

k = 20
centroids, W, prediction, score = classifierRBFN(pca_train, pca_test, amount_pcs, k_centroids=k)
