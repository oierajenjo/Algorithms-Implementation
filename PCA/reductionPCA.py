# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def reductionPCA(dataset):
    # Preprocessing
    # dataset = dataset.loc[dataset['HeatLoad'] == 10000]
    X = dataset.drop(['Test_nr', 'Faulty', 'HeatLoad'], 1)
    Y = dataset[['Test_nr', 'Faulty', 'HeatLoad']]
    # print("X")
    # print(X)
    # print("Y")
    # print(Y)

    # # Splitting the dataset into the Training set and Test set
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

    # STANDARDISE
    sc = StandardScaler()
    X = sc.fit_transform(X)
    # X_test = sc.transform(X_test)

    # Applying PCA
    pca = PCA()
    PC_X = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    columns = ['PC ' + str(i) for i in range(1, len(explained_variance)+1)]
    PC_X = pd.DataFrame(data=PC_X, columns=columns)
    finalX = pd.concat([PC_X, Y], axis=1)

    # PLOTTING
    # fig = plt.figure(figsize=(8, 8))
    # # ax = plt.axes(projection='3d')
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel('Principal Component 1', fontsize=15)
    # ax.set_ylabel('Principal Component 2', fontsize=15)
    # # ax.set_zlabel('Principal Component 3', fontsize=15)
    # ax.set_title('3 component PCA', fontsize=20)
    # targets = [0, 1]
    # colors = ['g', 'r']
    # for target, color in zip(targets, colors):
    #     indicesToKeep = finalX['Faulty'] == target
    #     ax.scatter(finalX.loc[indicesToKeep, 'principal component 1'],
    #                finalX.loc[indicesToKeep, 'principal component 2'],
    #                # finalX.loc[indicesToKeep, 'principal component 3'],
    #                c=color)
    #
    # ax.legend(['Not Faulty', 'Faulty'])
    # ax.grid()
    # fig.show()
    # fig.savefig('data/foo.png')


    print(PC_X)
    # print(X_test)
    print("Explained Variance")
    print(explained_variance)

    return PC_X, Y, explained_variance, finalX
