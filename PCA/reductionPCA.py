# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def standardise(X):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X


def reductionPCA(X, accuracy):
    # Amount of PCs to keep
    # Applying PCA
    pca = PCA()
    pca.fit_transform(X)
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
    fig.savefig('results/PCs(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')

    val = 0
    amount_pcs = 0
    for ev in explained_variance:
        val += ev
        amount_pcs += 1
        if val >= accuracy:
            break
    print(val)
    print(amount_pcs)

    # Applying PCA
    pca = PCA(n_components=amount_pcs)
    pc_x = pca.fit_transform(X)
    # ex_var = pca.explained_variance_ratio_

    return pc_x, explained_variance, amount_pcs


def plotting(pca):
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
        indicesToKeep = pca['Faulty'] == target
        ax.scatter3D(pca.loc[indicesToKeep, 'PC1'],
                     pca.loc[indicesToKeep, 'PC2'],
                     pca.loc[indicesToKeep, 'PC3'],
                     c=color,
                     marker=m)
    ax.grid()
    return fig, ax
