# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
from time import process_time

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def standardise(X):
    sc = StandardScaler()
    s_X = sc.fit_transform(X)
    return s_X


def reductionPCA(X, accuracy):
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
    fig.savefig('results/PCs(' + str(accuracy) + ').png')

    # Amount of PCs to keep
    val = 0
    amount_pcs = 0
    for ev in explained_variance:
        val += ev
        amount_pcs += 1
        if val >= accuracy:
            break
    print("Accuracy with " + str(amount_pcs) + " amount of centroids: " + str(val))

    # Applying PCA
    t1_start = process_time()
    pca = PCA(n_components=amount_pcs)
    pc_x = pca.fit_transform(X)
    # ex_var = pca.explained_variance_ratio_
    t1_stop = process_time()
    f = open("results/PCAResults(" + str(accuracy) + "-" + str(amount_pcs) + ").txt", 'w+')
    f.write(
        "%d amount of PCs, %.4f accuracy, not noisy + 2 noisy data tests\nAll test samples together\nTime needed for "
        "all PCA in seconds: %.7f\n"
        % (amount_pcs, accuracy, t1_stop - t1_start))
    f.write("Time needed for one PCA sample in seconds: %.7f\n " % ((t1_stop - t1_start)/len(pc_x)))
    f.close()
    return pc_x, explained_variance, amount_pcs


def plotting(pca):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    targets = [0, 1]
    colors = ['g', 'r']
    markers = ['*', '*']
    for target, color, m in zip(targets, colors, markers):
        indicesToKeep = pca['Faulty'] == target
        ax.scatter3D(pca.loc[indicesToKeep, pca.columns[0]],
                     pca.loc[indicesToKeep, pca.columns[1]],
                     pca.loc[indicesToKeep, pca.columns[2]],
                     c=color,
                     marker=m)
    ax.grid()
    return fig, ax
