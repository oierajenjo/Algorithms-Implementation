# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
import numpy as np
import os.path
from os import path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import Resources.retrieveData as retrieveData

path_parent = os.path.dirname(os.getcwd())
os.chdir(path_parent)
print(os.getcwd())

if not path.exists(os.getcwd()+'\Resources\\allData.csv'):
    print(1)
    os.system(os.getcwd()+'\Resources\\retrieveData.py')
    print(2)


# # Importing Dataset
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# dataset = pd.read_csv(url, names=names)
# dataset.head()
# # print("DATASET")
# # print(dataset)
#
# # Preprocessing
# X = dataset.drop('Class', 1)
# y = dataset['Class']
#
# # Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# # STANDARDISE
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# # Applying PCA
# pca = PCA()
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
#
# explained_variance = pca.explained_variance_ratio_
#
# # # 1 principal component to train our algorithm
# # pca = PCA(n_components=1)
# # X_train = pca.fit_transform(X_train)
# # X_test = pca.transform(X_test)
#
#
# print(X_train)
# print(X_test)
# print(explained_variance)
