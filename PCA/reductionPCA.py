# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def reductionPCA(dataset, conditions):
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
    print(X)
    print("Y")
    print(Y)

    # STANDARDISE
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Applying PCA
    pca = PCA()
    PC_X = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    columns = ['PC' + str(i) for i in range(1, len(explained_variance) + 1)]
    PC_X = pd.DataFrame(data=PC_X, columns=columns)
    finalX = pd.concat([PC_X, Y], axis=1)
    print("Final X")
    print(finalX.loc[finalX['Faulty'] == 0].tail())

    # PLOTTING
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    # ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component PCA', fontsize=20)
    targets = [0, 1]
    colors = ['g', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = finalX['Faulty'] == target
        ax.scatter3D(finalX.loc[indicesToKeep, 'PC1'],
                     finalX.loc[indicesToKeep, 'PC2'],
                     finalX.loc[indicesToKeep, 'PC3'],
                     c=color)

    ax.legend(['Not Faulty', 'Faulty'])
    ax.grid()
    fig.show()
    fig.savefig('results/3d.png')

    print(PC_X)
    print("Explained Variance")
    print(explained_variance)

    fig, ax = plt.subplots()
    ax.bar(columns, explained_variance)
    plt.title("Principal Components")
    plt.show()
    fig.savefig(os.getcwd() + '/results/PC.png')

    return PC_X, Y, explained_variance, finalX
