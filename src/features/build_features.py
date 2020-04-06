import os
import sys
sys.path.append('.')
import click
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
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
    X_train = one_hot_encode_categorical_features(X_train)
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

def one_hot_encode_categorical_features(X_train):
    """One-hot encodes the categorical features, adds these to the
    training data, then drops the original columns. Returns the
    transformed X_train data as a pandas Dataframe.
    """
    cols_to_one_hot_encode = X_train.dtypes[X_train.dtypes == 'object'].index

    ohe = OneHotEncoder(drop='first', sparse=False)
    ohe.fit(X_train[cols_to_one_hot_encode])

    ohe_features = ohe.transform(X_train[cols_to_one_hot_encode])
    ohe_feature_names = ohe.get_feature_names(cols_to_one_hot_encode)
    ohe_df = pd.DataFrame(
        ohe_features,
        columns=ohe_feature_names
    )

    X_train = X_train.assign(**ohe_df)
    X_train = X_train.drop(columns=cols_to_one_hot_encode)

    return X_train

def transform_target(y_train):
    """Transform target into zeros and ones for modeling
    """
    y_train['Churn'] = y_train['Churn'].map({'Yes': 1, 'No': 0})

    return y_train

if __name__ == "__main__":
    cli()