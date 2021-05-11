# https://www.hackerearth.com/blog/developers/radial-basis-function-network
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from PCA.reductionPCA import plotting


def classifierRBFN(pca_train, pca_test, amount_pcs, k_centroids=8):
    X_train = pca_train.iloc[:, :amount_pcs].to_numpy()
    X_test = pca_test.iloc[:, :amount_pcs].to_numpy()
    Y_train = pca_train['Faulty'].to_numpy()
    Y_test = pca_test['Faulty'].to_numpy()

    # Y_train = pca_train.iloc[:, amount_pcs:]
    # Y_test = pca_test.iloc[:, amount_pcs:]
    # Y_train['NFaulty'] = np.logical_xor(Y_train['Faulty'], 1).astype(int)
    # Y_test['NFaulty'] = np.logical_xor(Y_test['Faulty'], 1).astype(int)
    # Y_test = Y_test[['Faulty', 'NFaulty']].to_numpy()
    # Y_train = Y_train[['Faulty', 'NFaulty']].to_numpy()

    # Get centroids for RBF
    centroids, k_centroids = get_centroids(X_train, k_cent=k_centroids)

    # Plotting centroids
    fig, ax = plotting(pca_train)
    ax.set_title('Centroids', fontsize=20)
    ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker="x", color='k')
    fig.show()
    fig.savefig('results/Centroids(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')

    # Standard deviation
    sigma = get_sigma(centroids)

    # Train the RBFN H matrix
    H_train = get_H_matrix(X_train, centroids, sigma)
    print("H_train")
    print(H_train)

    # Get the Weights
    W = get_weights(H_train, Y_train)
    print("W")
    print(W)

    # Test the RBFN H matrix
    H_test = get_H_matrix(X_test, centroids, sigma)
    print("H_test")
    print(H_test)

    prediction, score = make_prediction(H_test, W, Y_test)
    print("Prediction")
    print(prediction)
    print("Accuracy: " + str(score) + "%")

    return centroids, W, prediction, score


def get_centroids(X, k_cent):
    # KMeans algorithm
    km = KMeans(n_clusters=k_cent).fit(X)

    cent = np.array(km.cluster_centers_)
    print("Centroids")
    print(cent)
    return cent, k_cent


def get_sigma(cent):
    k_cent = len(cent)
    m = 0
    for i in range(k_cent):
        for j in range(k_cent):
            d = np.linalg.norm(cent[i] - cent[j])
            if d > m:
                m = d
    d = m
    sigma = d / np.sqrt(2 * k_cent)
    return sigma


def get_H_matrix(X, cent, sigma):
    shape = X.shape
    row = shape[0]
    column = len(cent)
    H = np.empty((row, column), dtype=float)
    for p in range(row):
        for m in range(column):
            dist = np.linalg.norm(X[p] - cent[m])
            H[p][m] = np.exp(-(dist.T * dist) / np.power(2 * sigma, 2))  # CHECK THIS SIGMA
            # H[p][m] = np.exp(-np.pow(dist, 2) / np.pow(2 * sigma, 2))
    return H


def get_weights(H, Y):
    HTH = np.dot(H.T, H)
    HTH_inv = np.linalg.inv(HTH)
    fac = np.dot(HTH_inv, H.T)
    W = np.dot(fac, Y)
    return W


def make_prediction(H, W, Y):
    prediction = np.dot(H, W)
    prediction = 0.5 * (np.sign(prediction - 0.5) + 1)
    score = accuracy_score(prediction, Y) * 100
    return prediction, score
