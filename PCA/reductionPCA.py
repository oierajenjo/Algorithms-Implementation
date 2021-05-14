# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
import os
import re
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def reset_ind(X):
    X.reset_index(inplace=True)
    X = X.drop('index', axis=1)
    return X


def to_code(string):
    code = "(" + string + ")"
    code = re.sub("[a-zA-Z_]+", lambda m: "df['%s']" % m.group(0), code)
    code = code.replace(",", ") & (")
    code = code.replace(";", " | ")
    code = code.replace("=", "==")
    code = code.replace(">==", ">=")
    code = code.replace("<==", "<=")
    return code


def save_csv_file(name, df):
    df.to_csv(os.getcwd() + name, index=False, header=True)


def read_csv_data(file, conditions=None):
    df = pd.read_csv(file)
    df.head()
    if conditions is not None:
        print("In")
        df = df.loc[to_code(conditions)]
    X, Y = preprocess_data(df)
    return X, Y


def set_train_validationData(X, Y):
    # Set Test and training data
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.01, random_state=42)

    # Remove previous index
    X_train = reset_ind(X_train)
    Y_train = reset_ind(Y_train)
    X_val = reset_ind(X_val)
    Y_val = reset_ind(Y_val)

    save_csv_file('\data\\trainData.csv', pd.concat([X_train, Y_train], axis=1))
    save_csv_file('\data\\validationData.csv', pd.concat([X_val, Y_val], axis=1))

    return X_train, Y_train, X_val, Y_val


def preprocess_data(df):
    X = df.drop(['Test_nr', 'Faulty', 'HeatLoad', 'T_set', 'Cpr_Scale'], axis=1)
    Y = df[['Test_nr', 'Faulty', 'HeatLoad', 'T_set', 'Cpr_Scale']]
    X = reset_ind(X)
    Y = reset_ind(Y)
    # print("X")
    # print(X.head())
    # print("Y")
    # print(Y.head())
    return X, Y


def reductionPCA(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, accuracy):
    # STANDARDISE
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_validation = sc.transform(X_validation)
    X_test = sc.transform(X_test)

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
    # fig.savefig('results/PC(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')

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
    pc_x_train = pca.fit_transform(X_train)
    pc_x_validation = pca.transform(X_validation)
    pc_x_test = pca.transform(X_test)
    # ex_var = pca.explained_variance_ratio_

    # PC DataFrame
    columns = ['PC' + str(i) for i in range(1, amount_pcs + 1)]

    pc_x_train = pd.DataFrame(data=pc_x_train, columns=columns)
    pca_train = pd.concat([pc_x_train, Y_train], axis=1)

    pc_x_validation = pd.DataFrame(data=pc_x_validation, columns=columns)
    pca_validation = pd.concat([pc_x_validation, Y_validation], axis=1)

    pc_x_test = pd.DataFrame(data=pc_x_test, columns=columns)
    pca_test = pd.concat([pc_x_test, Y_test], axis=1)

    print("Final PCA X")
    print(pca_train)

    # # PLOTTING
    # fig, ax = plotting(pca_train)
    # ax.set_title('3 component PCA', fontsize=20)
    # ax.legend(['Not Faulty', 'Faulty'])
    # fig.show()
    # fig.savefig('results/3d(' + datetime.today().strftime("%d%m%Y_%H-%M.%S") + ').png')

    return pca_train, pca_test, pca_validation, explained_variance, amount_pcs


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
    ax.grid()
    return fig, ax
