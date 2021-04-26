import os
from os import path
import pandas as pd
from PCA.reductionPCA import reductionPCA
from matplotlib import pyplot as plt

os.system('PCA\\reductionPCA.py')

file = os.getcwd() + '\data\\allData.csv'

if not path.exists(file):
    os.system(os.getcwd() + '\data\\retrieveData.py')

# Importing Dataset
dataset = pd.read_csv(file)
dataset.head()
# print("DATASET")
# print(dataset)

PC_X, Y, explained_variance = reductionPCA(dataset)

plt.bar(explained_variance)


