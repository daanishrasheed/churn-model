import os
import sys
sys.path.append('.')
import click
import pandas as pd
from src.localpaths import *
from src.data.make_dataset import load_training_data

@click.group()
def cli():
    pass

def featurize_X_train(X_train):
    """Applies all featurization steps to X_train
    """
    X_train = drop_customer_id(X_train)
    X_train['gender']=X_train['gender'].map({'Female': 1, 'Male': 0})
    X_train['Partner']=X_train['Partner'].map({'Yes': 1, 'No': 0})
    X_train['Dependents']=X_train['Dependents'].map({'Yes': 1, 'No': 0})
    X_train['PhoneService']=X_train['PhoneService'].map({'Yes': 1, 'No': 0})
    X_train['PaperlessBilling']=X_train['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    return X_train


@cli.command()
def create_featurized_data():
    """Creates X and y training set files for modeling
    Saves the data to data/processed
    """
    print('loading data')
    X_train, y_train = load_training_data(clean=True)

    print('featurizing data')
    X_train = featurize_X_train(X_train)
    y_train = transform_target(y_train)

    print('saving data')
    X_train.to_csv(X_TRAIN_FEATURIZED_PATH, index=False)
    y_train.to_csv(Y_TRAIN_FEATURIZED_PATH, index=False)


def drop_customer_id(X_train):
    """Drops the column from Customer ID
    """
    X_train = X_train.drop(columns=['customerID'])

    return X_train

def transform_target(y_train):
    """Transform target into zeros and ones for modeling
    """
    y_train['Churn'] = y_train['Churn'].map({'Yes': 1, 'No': 0})

    return y_train

if __name__ == "__main__":
    cli()