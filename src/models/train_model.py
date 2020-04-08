import os
import pickle
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score


from src.data.make_dataset import load_training_data
from src.localpaths import *


def store_model_and_results(model, X_train, y_train):
    """Saves model evaluation metrics to /models/model_results.csv, and
    saves the pickled model to /models.
    
    """
    model_results_filepath = os.path.join(MODELS_DIRECTORY, 'model_results.csv')
    model_filename = str(hash(np.random.rand()))+ '.pkl'
    model_string = str(model)
    
    accuracy = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring='accuracy'))
    precision = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring='accuracy'))
    recall = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring='accuracy'))
    f1 = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring='accuracy'))
    roc_auc = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring='accuracy'))

    data_to_save = {
        'model_filename': [model_filename],
        'model_string': [model_string],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1': [f1],
        'roc_auc': [roc_auc]
    }
    df_results = pd.read_csv(model_results_filepath)

    print('fitting model before pickling')
    model.fit(X_train, y_train)

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


def print_model_results(model, X_train, y_train):
    """Prints model evaluation metrics
    """

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in metrics:
        metric_value = np.mean(cross_val_score(model, X_train, y_train['Churn'], cv=5, scoring=metric))
        print(f'{metric}: {metric_value:.2f}')





if __name__ == "__main__":
    pass