# https://www.hackerearth.com/blog/developers/radial-basis-function-network
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def get_XY(pca, amount_pcs):
    X = pca.iloc[:, :amount_pcs].to_numpy()
    # Y = pca.iloc[:, amount_pcs:]
    # Y['NFaulty'] = np.logical_xor(Y['Faulty'], 1).astype(int)
    # Y = Y[['Faulty', 'NFaulty']].to_numpy()
    Y = pca['Faulty'].to_numpy()
    return X, Y


def train_data(X_train, Y_train, k_centroids=8):
    # Get centroids for RBF
    centroids = get_centroids(X_train, k_cent=k_centroids)

    # Standard deviation
    sigma = get_sigma(centroids)

    # Train the RBFN H matrix
    H_train = get_H_matrix(X_train, centroids, sigma)
    # print("H_train")
    # print(H_train)

    # Get the Weights
    W = get_weights(H_train, Y_train)
    # print("W")
    # print(W)

    return centroids, W, sigma


def get_centroids(X, k_cent):
    # KMeans algorithm
    km = KMeans(n_clusters=k_cent).fit(X)

    cent = np.array(km.cluster_centers_)
    # print("Centroids")
    # print(cent)
    return cent


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
    rows = shape[0]
    columns = len(cent)
    H = np.empty((rows, columns), dtype=float)
    for p in range(rows):
        for m in range(columns):
            dist = np.linalg.norm(X[p] - cent[m])
            H[p][m] = np.exp(-(dist.T * dist) / np.power(2 * sigma, 2))  # CHECK THIS SIGMA
            # H[p][m] = np.exp(-np.pow(dist, 2) / np.pow(2 * sigma, 2))
    return H


def get_weights(H, Y):
    HTH = np.dot(H.T, H)
    HTH_inv = np.linalg.inv(HTH)
    temp = np.dot(HTH_inv, H.T)
    W = np.dot(temp, Y)
    return W


def measure_accuracy(X, Y, centroids, sigma, W):
    H = get_H_matrix(X, centroids, sigma)

    prediction = np.dot(H, W)
    prediction = 0.5 * (np.sign(prediction - 0.5) + 1)
    score = accuracy_score(prediction, Y) * 100
    # print("Prediction")
    # print(prediction)
    print("Accuracy: " + str(score) + "%")
    return score


def amount_centroids(pca_train, pca_validation, amount_pcs, k_max):
    X_pca_train, Y_pca_train = get_XY(pca_train, amount_pcs)
    X_pca_validation, Y_pca_validation = get_XY(pca_validation, amount_pcs)

    scores = []
    k_all = []
    for k in range(2, k_max + 1):
        print("Centroids: " + str(k))
        centroids, W, sigma = train_data(X_pca_train, Y_pca_train, k_centroids=k)

        # # Plotting centroids
        # fig, ax = plotting(pca_train)
        # ax.set_title('Centroids', fontsize=20)
        # ax.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker="x", color='k')
        # ax.legend(['Not Faulty', 'Faulty', 'Centroids'])
        # fig.show()
        # fig.savefig('results/Centroids(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')

        score = measure_accuracy(X_pca_validation, Y_pca_validation, centroids, sigma, W)
        scores.append(score)
        k_all.append(k)
        # if len(scores) >= 5:
        #     if scores[-5] == 100:
        #         break

    return scores, k_all
