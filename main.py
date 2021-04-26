import os
from os import path
import pandas as pd
from PCA.reductionPCA import reductionPCA
from matplotlib import pyplot as plt

file = os.getcwd() + '\data\\allData.csv'

if not path.exists(file):
    os.system(os.getcwd() + '\data\\retrieveData.py')

# Importing Dataset
dataset = pd.read_csv(file)
dataset.head()
# print("DATASET")
# print(dataset)
conditions = []
PC_X, Y, explained_variance, finalX = reductionPCA(dataset, conditions)

val = 0
n = 0
for ev in explained_variance:
    val += ev
    n += 1
    if val >= 0.995:
        break
print(val)
print(n)
