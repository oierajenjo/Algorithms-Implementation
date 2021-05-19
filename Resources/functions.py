import errno
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from Resources.retrieveData import download_file_from_google_drive


def retrieve_files(root, file_all, file_noisy, file_noisy2):
    if not os.path.exists(root + file_all):
        download_file_from_google_drive('allData.csv')

    if not os.path.exists(root + file_noisy):
        download_file_from_google_drive('allNoisyData.csv')

    if not os.path.exists(root + file_noisy2):
        download_file_from_google_drive('allNoisyData2.csv')


def create_folders():
    try:
        os.makedirs('results')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs('results/3D')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs('results/centroids')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


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


def set_train_validation_testData(X_all, Y_all, test_val_size):
    # Set Test and training data
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(X_all, Y_all, test_size=test_val_size, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5, random_state=42)

    # Remove previous index
    X_train = reset_ind(X_train)
    Y_train = reset_ind(Y_train)
    X_val = reset_ind(X_val)
    Y_val = reset_ind(Y_val)
    X_test = reset_ind(X_test)
    Y_test = reset_ind(Y_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


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
