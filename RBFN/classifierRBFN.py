# clustering dataset
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

from PCA.reductionPCA import plotting


def classifierRBFN(pc_x_train, pca_train, pca_test, amount_pcs):
    centroids = get_centroids(pca_train, amount_pcs)


def get_centroids(pca_train, amount_pcs):
    X = pca_train.iloc[:, list(range(amount_pcs))].to_numpy()
    print("X")
    print(X)

    # KMeans algorithm
    # K = 100
    kmeans_model = KMeans().fit(X)

    centroids = np.array(kmeans_model.cluster_centers_)
    print("Centroids")
    print(centroids)

    fig, ax = plotting(pca_train)
    ax.set_title('Centroids', fontsize=20)
    ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker="x", color='k')
    fig.show()
    return centroids
