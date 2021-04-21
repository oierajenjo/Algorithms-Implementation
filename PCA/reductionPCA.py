# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
import numpy as np
import os.path
from os import path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

path_parent = os.path.dirname(os.getcwd())
os.chdir(path_parent)
# print(os.getcwd())
file = os.getcwd() + '\Resources\\allData.csv'

if not path.exists(file):
    os.system(os.getcwd() + '\Resources\\retrieveData.py')

# Importing Dataset
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(file)
dataset.head()
# print("DATASET")
# print(dataset)

# Preprocessing
X = dataset.drop(['Test_nr', 'Faulty'], 1)
Y = dataset[['Test_nr', 'Faulty']]
# print("X")
# print(X)
# print("Y")
# print(Y)

# # Splitting the dataset into the Training set and Test set
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

# STANDARDISE
sc = StandardScaler()
X = sc.fit_transform(X)
# X_test = sc.transform(X_test)

# Applying PCA
pca = PCA(n_components=3)
X = pca.fit_transform(X)
# X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

print(X)
# print(X_test)

print("Explained Variance")
print(explained_variance)

