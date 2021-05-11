import os
from os import path
import pandas as pd
from PCA.reductionPCA import reductionPCA
from RBFN.classifierRBFN import classifierRBFN

file = os.getcwd() + '\data\\allData.csv'

if not path.exists(file):
    os.system(os.getcwd() + '\Resources\\retrieveData.py')

# Importing Dataset
dataset = pd.read_csv(file)
dataset.head()
# print("DATASET")
# print(dataset)
accuracy = 0.9995
pca_train, pca_test, explained_variance, amount_pcs = reductionPCA(dataset, accuracy)

k = 8
centroids, W, prediction, score = classifierRBFN(pca_train, pca_test, amount_pcs, k_centroids=k)
