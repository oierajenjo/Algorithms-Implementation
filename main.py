import os
from datetime import datetime
from time import process_time


from matplotlib import pyplot as plt

from PCA.reductionPCA import reductionPCA, plotting, standardise
from RBFN.classifierRBFN import train_data, measure_accuracy, get_XY, amount_centroids, plot_centroids
from Resources.functions import get_df, preprocess_data, read_csv_data, set_train_validation_testData, pca_df
from Resources.retrieveData import download_file_from_google_drive


def main():
    file_all = os.getcwd() + '/data/allData.csv'
    file_noisy = os.getcwd() + '/data/allNoisyData.csv'
    file_train = os.getcwd() + '/data/trainData.csv'
    file_validation = os.getcwd() + '/data/validationData.csv'
    file_test = os.getcwd() + '/data/testData.csv'

    """
    CHECK IF FILES EXIST
    If not execute download_file_from_google_drive
    """
    if not os.path.exists(file_all):
        download_file_from_google_drive('allData.csv')

    if not os.path.exists(file_noisy):
        download_file_from_google_drive('allNoisyData.csv')

    """
    SEPARATE DATA
    Data samples are separated in train, validation and test data
    """

    if not os.path.exists(file_validation) or not os.path.exists(file_train) or not os.path.exists(file_test):
        # Importing Dataset
        df = get_df(file_all)
        dfn = get_df(file_noisy)
        df.append(dfn)

        # Preprocessing
        X, Y = preprocess_data(df)
        """
        STANDARDISATION
        Standardisation process were data is standardised to mean 0 and var 1
        """
        X = standardise(X)

        """
        PCA
        If don't want to implement PCA comment this part
        """
        accuracy = 0.9995
        pc_X, explained_variance, amount_pcs = reductionPCA(X, accuracy)

        X_train, Y_train, X_validation, Y_validation, X_test, Y_test = set_train_validation_testData(pc_X, Y
                                                                                                     , amount_pcs)

    else:
        X_train, Y_train = read_csv_data(file_train)
        X_validation, Y_validation = read_csv_data(file_validation)
        X_test, Y_test = read_csv_data(file_test)

    amount_pcs = len(X_train.columns)
    print(amount_pcs)

    """
    PCA Matrix
    Generate 'PCA' matrix
    """
    pca_train = pca_df(X_train, Y_train)
    pca_validation = pca_df(X_validation, Y_validation)
    pca_test = pca_df(X_test, Y_test)
    print("Final PCA X")
    print(pca_train)

    # Plotting PCA training data
    fig, ax = plotting(pca_train)
    ax.set_title('3 component PCA Train', fontsize=20)
    ax.legend(['Non Faulty', 'Faulty'])
    fig.show()
    fig.savefig('results/3d(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')

    """
    RBFN
    Instead of validation use test data
    If don't want to implement RBFN comment this part
    """
    # Accuracy per amount of centroid
    # centroids_max = int(len(pca_train.index)/10)
    centroids_max = 100
    step = 1
    scores, c_all = amount_centroids(pca_train, pca_validation, amount_pcs, centroids_max, step)

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
    Accuracy with noisy data
    """
    X_pca_train, Y_pca_train = get_XY(pca_train, amount_pcs)
    centroids, W, sigma = train_data(X_pca_train, Y_pca_train, n_centroids=centroids_max)

    # Plotting centroids
    plot_centroids(pca_train, centroids)

    t1_start = process_time()
    X_pca_test, Y_pca_test = get_XY(pca_test, amount_pcs)
    score = measure_accuracy(X_pca_test, Y_pca_test, centroids, sigma, W)
    t1_stop = process_time()
    print("Testing time in seconds:", t1_stop - t1_start)

    fig, ax = plotting(pca_test)
    ax.set_title('3 component PCA Test', fontsize=20)
    ax.legend(['Non Faulty', 'Faulty'])
    fig.show()
    fig.savefig('results/3d(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')
    plot_centroids(pca_test, centroids)


if __name__ == "__main__":
    main()
