import os
import pandas as pd
from sklearn.model_selection import train_test_split


def set_train_validation_testData(pc_x, Y, amount_pcs):
    columns = ['PC' + str(i) for i in range(1, amount_pcs + 1)]
    X = pd.DataFrame(data=pc_x, columns=columns)

    # Set Test and training data
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.01, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.01, random_state=42)

    # Remove previous index
    X_train = reset_ind(X_train)
    Y_train = reset_ind(Y_train)
    X_val = reset_ind(X_val)
    Y_val = reset_ind(Y_val)
    X_test = reset_ind(X_test)
    Y_test = reset_ind(Y_test)

    save_csv_file('/data/trainData.csv', pd.concat([X_train, Y_train], axis=1))
    save_csv_file('/data/validationData.csv', pd.concat([X_val, Y_val], axis=1))
    save_csv_file('/data/testData.csv', pd.concat([X_test, Y_test], axis=1))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def get_df(file):
    df = pd.read_csv(file)
    df.head()
    df = df.loc[(df['HeatLoad'] == 10000) & ((df['T_set'] == 0) | (df['T_set'] == 12))
                & ((df['Cpr_Scale'] >= 0.9) | (df['Cpr_Scale'] == 0.3))]
    return df


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


def read_csv_data(file, conditions=None):
    df = pd.read_csv(file)
    df.head()
    # if conditions is not None:
    #     print("In")
    #     df = df.loc[to_code(conditions)]
    X, Y = preprocess_data(df)
    return X, Y


# PCA DataFrame
def pca_df(pc_x, Y):
    pca = pd.concat([pc_x, Y], axis=1)
    return pca


# def to_code(string):
#     code = "(" + string + ")"
#     code = re.sub("[a-zA-Z_]+", lambda m: "df['%s']" % m.group(0), code)
#     code = code.replace(",", ") & (")
#     code = code.replace(";", " | ")
#     code = code.replace("=", "==")
#     code = code.replace(">==", ">=")
#     code = code.replace("<==", "<=")
#     return code

def reset_ind(X):
    X.reset_index(inplace=True)
    X = X.drop('index', axis=1)
    return X


def save_csv_file(name, df):
    df.to_csv(os.getcwd() + name, index=False, header=True)
