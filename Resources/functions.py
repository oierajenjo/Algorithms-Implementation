import os
import pandas as pd
from sklearn.model_selection import train_test_split


def get_df(file):
    d = pd.read_csv(file)
    d.head()
    d = d.loc[(d['HeatLoad'] == 10000) & ((d['T_set'] == 0) | (d['T_set'] == 12))
              & ((d['Cpr_Scale'] >= 0.9) | (d['Cpr_Scale'] == 0.3))]
    d = reset_ind(d)
    return d


def preprocess_data(df):
    X = df.drop(['Test_nr', 'Faulty', 'HeatLoad', 'T_set', 'Cpr_Scale'], axis=1)
    Y = df[['Test_nr', 'Faulty', 'HeatLoad', 'T_set', 'Cpr_Scale']]
    X = reset_ind(X)
    Y = reset_ind(Y)
    return X, Y


def set_train_validation_testData(pc_X, Y_all, amount_pcs):
    columns = ['PC' + str(i) for i in range(1, amount_pcs + 1)]
    X_all = pd.DataFrame(data=pc_X, columns=columns)

    # Set Test and training data
    X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=0.01, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.01, random_state=42)

    # Remove previous index
    X_train = reset_ind(X_train)
    Y_train = reset_ind(Y_train)
    X_val = reset_ind(X_val)
    Y_val = reset_ind(Y_val)
    X_test = reset_ind(X_test)
    Y_test = reset_ind(Y_test)

    return X_all, X_train, Y_train, X_val, Y_val, X_test, Y_test


def read_csv_data(file):
    df = pd.read_csv(file)
    df.head()
    X, Y = preprocess_data(df)
    return X, Y


# PCA DataFrame
def pca_df(pc_x, Y):
    pca = pd.concat([pc_x, Y], axis=1)
    return pca


def reset_ind(X):
    X.reset_index(inplace=True)
    X = X.drop('index', axis=1)
    return X


def save_csv_file(name, df):
    df.to_csv(os.getcwd() + name, index=False, header=True)
