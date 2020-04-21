import os
import pickle
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import click

from src.data.make_dataset import load_training_data, clean_X
from src.localpaths import *


@click.group()
def cli():
    pass

@cli.command()
@click.option('--file-name', type=str, required=True)
def predict(file_name):
    """Predicts 'churn' or 'not churn' for each row of data
    in file_name. The file must be comma delimited. Column
    names don't matter, but the order of the columns should 
    be the same as the order of the original training data.
    """
    # Load data
    X = pd.read_csv(file_name)

    # Clean and featurize our data
    X = clean_X(X)
    X = featurize_X(X)

    # Make predictions

    # Print those predictions
    print(X)




if __name__ == "__main__":
    cli()