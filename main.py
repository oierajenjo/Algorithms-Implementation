import os
from datetime import datetime
from time import process_time

import pandas as pd
from matplotlib import pyplot as plt

from PCA.reductionPCA import reductionPCA, plotting, standardise
from RBFN.classifierRBFN import train_data, measure_accuracy, get_XY, amount_centroids, plot_centroids
from Resources.functions import get_df, preprocess_data, read_csv_data, set_train_validation_testData, pca_df, \
    save_csv_file
from Resources.retrieveData import download_file_from_google_drive

accuracy = 0.995  # Accuracy for the PCA

root = os.getcwd()
file_all = '/data/allData.csv'
file_noisy = '/data/allNoisyData.csv'
file_merged = '/data/mergedData(' + str(accuracy) + ').csv'
file_train = '/data/trainData(' + str(accuracy) + ').csv'
file_val = '/data/validationData(' + str(accuracy) + ').csv'
file_test = '/data/testData(' + str(accuracy) + ').csv'

"""
CHECK IF FILES EXIST
If not execute download_file_from_google_drive
"""
if not os.path.exists(root + file_all):
    download_file_from_google_drive('allData.csv')

if not os.path.exists(root + file_noisy):
    download_file_from_google_drive('allNoisyData.csv')

if not os.path.exists(root + file_val) or not os.path.exists(root + file_train) \
        or not os.path.exists(root + file_test):

    # Importing Datasets and group them
    df = get_df(root + file_all)
    print("Number of non-noisy samples: " + str(len(df.index)))
    dfn = get_df(root + file_noisy)
    print("Number of noisy samples: " + str(len(dfn.index)))
    df_all = pd.concat([df, dfn], ignore_index=True)

    # Preprocessing
    X, Y = preprocess_data(df_all)

    """
    STANDARDISATION
    Standardisation process were data is standardised to mean 0 and var 1
    """
    s_X = standardise(X)

    """
    PCA
    If don't want to implement PCA comment this part
    """
    pc_X, explained_variance, amount_pcs = reductionPCA(s_X, accuracy)

    """
    SEPARATE DATA
    Data samples are separated in train, validation and test data
    """
    X_all, X_train, Y_train, X_val, Y_val, X_test, Y_test = set_train_validation_testData(pc_X, Y, amount_pcs)

    save_csv_file(file_train, pd.concat([X_train, Y_train], axis=1))
    save_csv_file(file_val, pd.concat([X_val, Y_val], axis=1))
    save_csv_file(file_test, pd.concat([X_test, Y_test], axis=1))
    save_csv_file(file_merged, pd.concat([X_all, Y], axis=1))
else:
    X_all, Y = read_csv_data(root + file_merged)
    X_train, Y_train = read_csv_data(root + file_train)
    X_val, Y_val = read_csv_data(root + file_val)
    X_test, Y_test = read_csv_data(root + file_test)

amount_pcs = len(X_train.columns)
print(amount_pcs)

"""
PCA Matrix
Generate 'PCA' matrix
"""
pca_all = pca_df(X_all, Y)
pca_train = pca_df(X_train, Y_train)
pca_validation = pca_df(X_val, Y_val)
pca_test = pca_df(X_test, Y_test)
print("Training  PCA")
print(pca_train)

# Plotting PCA training data
fig, ax = plotting(pca_all)
ax.set_title('3 component PCA All', fontsize=20)
ax.legend(['Non Faulty', 'Faulty'])
fig.show()
fig.savefig('results/3d(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')

"""
RBFN
Instead of validation use test data
If don't want to implement RBFN comment this part
"""
# Accuracy per amount of centroid
centroids_max = 100
step = 1
init_cent = 2
scores, c_all = amount_centroids(pca_train, pca_validation, amount_pcs, centroids_max, step=step)

# Plotting accuracy per Amount of Centroids
plt.plot(c_all, scores)
plt.title('Accuracy per Amount of Centroids')
plt.xlabel('Amount of Centroids')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
fig.savefig('results/Accuracy-Centroids(' + str(c_all[0]) + '-' + str(c_all[-1]) + ').png')

"""
RBFN
Accuracy with Test data
"""
# dfs = pd.read_csv(root + '/results/score_cent(2-100).csv')
# dfs.head()
# c_all = dfs['centroids'].tolist()
# scores = dfs['scores'].tolist()

# Train
X_pca_train, Y_pca_train = get_XY(pca_train, amount_pcs)
n_cent = c_all[scores.index(max(scores))]
centroids, W, sigma = train_data(X_pca_train, Y_pca_train, n_centroids=n_cent)


# Test
t1_start = process_time()
X_pca_test, Y_pca_test = get_XY(pca_test, amount_pcs)
score = measure_accuracy(X_pca_test, Y_pca_test, centroids, sigma, W)
t1_stop = process_time()
print("Testing time in seconds:" + str(t1_stop - t1_start) + "s")
print("Score: " + str(score) + "%")


# Plots
fig, ax = plotting(pca_test)
ax.set_title('3 component PCA Test', fontsize=20)
ax.legend(['Non Faulty', 'Faulty'])
fig.show()
fig.savefig('results/3d(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')

# Plotting centroids
plot_centroids(pca_train, centroids, 'results/' + str(n_cent) + 'CentroidsTrain-Optimal.png')
plot_centroids(pca_test, centroids, 'results/' + str(n_cent) + 'CentroidsTest-Optimal.png')
