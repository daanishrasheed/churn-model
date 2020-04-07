import os
import pickle
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.make_dataset import load_training_data
from src.localpaths import *


def store_model_and_results(model, X_train, y_train):
    """Saves model evaluation metrics to /models/model_results.csv, and
    saves the pickled model to /models.
    
    """
    model_results_filepath = os.path.join(MODELS_DIRECTORY, 'model_results.csv')
    model_filename = str(hash(np.random.rand()))+ '.pkl'
    model_string = str(model)

    X, X_validation, y, y_validation = train_test_split(X_train, y_train, test_size = 0.25, random_state=42)

    model.fit(X, y)
    accuracy = model.score(X_validation, y_validation)

    data_to_save = {
        'model_filename': [model_filename],
        'model_string': [model_string],
        'accuracy': [accuracy]
    }
    df_results = pd.read_csv(model_results_filepath)

    print(f"saving pickled model to {model_filename}")
    with open(os.path.join(MODELS_DIRECTORY, model_filename), 'wb') as f:
        pickle.dump(model, f)

    if os.path.exists(model_results_filepath):
        print('writing model results to existing results CSV files')
        df_results = pd.read_csv(model_results_filepath)
        new_results = pd.DataFrame(data_to_save)
        df_results = df_results.append(new_results, ignore_index=True)
        df_results.to_csv(model_results_filepath, index=False)
    else:
        print('model results file does not exist -- creating new model results CSV file and writing results')
    df_results.to_csv(model_results_filepath, index=False)

if __name__ == "__main__":
    pass