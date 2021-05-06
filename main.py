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
pc_x_train, Y_train, pc_x_test, Y_test , explained_variance, pca_train, pca_test = reductionPCA(dataset, accuracy)

n = len(pc_x_train.columns)
classifierRBFN(pc_x_train, pca_train, pca_test, n)
