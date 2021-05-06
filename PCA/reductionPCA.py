# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def reductionPCA(dataset, accuracy):
    # Preprocessing
    dataset = dataset.loc[(dataset['HeatLoad'] == 10000) & ((dataset['T_set'] == 0) | (dataset['T_set'] == 12))
                          & ((dataset['Cpr_Scale'] >= 0.9) | (dataset['Cpr_Scale'] == 0.3))]
    X = dataset.drop(['Test_nr', 'Faulty', 'HeatLoad', 'T_set', 'Cpr_Scale'], axis=1)
    Y = dataset[['Test_nr', 'Faulty', 'HeatLoad', 'T_set', 'Cpr_Scale']]
    Y.reset_index(inplace=True)
    X.reset_index(inplace=True)
    X = X.drop('index', axis=1)
    Y = Y.drop('index', axis=1)
    print("X")
    print(X.head())
    print("Y")
    print(Y.head())

    # Set Test and training data
    X_tr, X_te, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=0)

    # STANDARDISE
    sc = StandardScaler()
    X_train = sc.fit_transform(X_tr)
    X_test = sc.transform(X_te)

    # Amount of PCs to keep
    # Applying PCA
    pca = PCA()
    pca.fit_transform(X_train)
    explained_variance = pca.explained_variance_ratio_
    print("Explained Variance")
    print(explained_variance)

    # Explained Variance Ratio PLOT
    fig, ax = plt.subplots()
    columns = ['PC' + str(i) for i in range(1, len(explained_variance) + 1)]
    ax.bar(columns, explained_variance)
    plt.title("Explained Variance per PC")
    plt.ylabel("Explained Variance Ratio")
    plt.xlabel("Principal Components")
    plt.show()
    fig.savefig(os.getcwd() + '/results/PC.png')

    val = 0
    n = 0
    for ev in explained_variance:
        val += ev
        n += 1
        if val >= accuracy:
            break
    print(val)
    print(n)

    # Applying PCA
    pca = PCA(n_components=n)
    pc_x_train = pca.fit_transform(X_train)
    pc_x_test = pca.transform(X_test)
    ex_var = pca.explained_variance_ratio_

    # PC DataFrame
    columns = ['PC' + str(i) for i in range(1, len(ex_var) + 1)]
    pc_x_train = pd.DataFrame(data=pc_x_train, columns=columns)
    pca_train = pd.concat([pc_x_train, Y_train.reset_index()], axis=1)

    pc_x_test = pd.DataFrame(data=pc_x_test, columns=columns)
    pca_test = pd.concat([pc_x_test, Y_test.reset_index()], axis=1)

    print("Final PCA X")
    print(pca_train)

    # PLOTTING
    fig, ax = plotting(pca_train)
    ax.set_title('3 component PCA', fontsize=20)
    fig.show()
    fig.savefig('results/3d.png')

    return pc_x_train, Y_train.reset_index(), pc_x_test, Y_test.reset_index(), explained_variance, pca_train, pca_test


def plotting(pca_train):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    # ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    targets = [0, 1]
    colors = ['g', 'r']
    markers = ['*', '*']
    for target, color, m in zip(targets, colors, markers):
        indicesToKeep = pca_train['Faulty'] == target
        ax.scatter3D(pca_train.loc[indicesToKeep, 'PC1'],
                     pca_train.loc[indicesToKeep, 'PC2'],
                     pca_train.loc[indicesToKeep, 'PC3'],
                     c=color,
                     marker=m)
    ax.legend(['Not Faulty', 'Faulty'])
    ax.grid()
    return fig, ax
