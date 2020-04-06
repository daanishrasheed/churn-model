import os
import sys
sys.path.append('.')
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from src.localpaths import *

@click.group()
def cli():
    pass

@cli.command()
def create_train_test_split():
    print('loading data')
    df = pd.read_csv(RAW_DATA_PATH)
    print('creating x and y')
    X=df.drop(columns=['Churn'])
    y=df[['Churn']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print('saving data files')
    X_train.to_csv(X_TRAIN_RAW_PATH, index = False)
    X_test.to_csv(X_TEST_RAW_PATH, index = False)
    y_train.to_csv(Y_TRAIN_RAW_PATH, index = False)
    y_test.to_csv(Y_TEST_RAW_PATH, index = False)

@cli.command()
def create_clean_training_data():
    """Reads in raw X and y training data, clean it, and writes clean
    training data out to the data/interim directory.
    """
    print('loading data')
    X_train, y_train = load_training_data()

    print('cleaning data')
    bad_values_idxs = X_train[X_train['TotalCharges']== ' '].index
    X_train.loc[bad_values_idxs, 'TotalCharges'] = 20
    X_train['TotalCharges']=X_train['TotalCharges'].astype(float)

    print('writing data files')
    X_train.to_csv(X_TRAIN_CLEAN_PATH, index=False)
    y_train.to_csv(Y_TRAIN_CLEAN_PATH, index=False)


def load_training_data(clean=False, final=False):
    """Return x_train and y_train
    """
    if clean:
        X_train = pd.read_csv(X_TRAIN_CLEAN_PATH)
        y_train = pd.read_csv(Y_TRAIN_CLEAN_PATH)
    elif final:
        X_train = pd.read_csv(X_TRAIN_FEATURIZED_PATH)
        y_train = pd.read_csv(Y_TRAIN_FEATURIZED_PATH)
    else:
        X_train=pd.read_csv(X_TRAIN_RAW_PATH)
        y_train=pd.read_csv(Y_TRAIN_RAW_PATH)

    return X_train, y_train

def load_test_data():
    """Return x_test and y_test
    """
    X_test=pd.read_csv(X_TEST_RAW_PATH)
    y_test=pd.read_csv(Y_TEST_RAW_PATH)

    return X_test, y_test


if __name__ == "__main__":
    cli()