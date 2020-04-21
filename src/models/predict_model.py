import os
import pickle
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import click

from src.data.make_dataset import load_training_data
from src.localpaths import *


@click.group()
def cli():
    pass

@cli.command()
@click.option('--file-name', type=str, required=True)
def predict(file_name):
    """Docstring
    
    """
    pass




if __name__ == "__main__":
    cli()