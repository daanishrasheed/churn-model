import os
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


from src.data.make_dataset import load_training_data
from src.localpaths import *



def plot_learning_curve(model, X_train, y_train, zoom_out=True):
    """Plots a learning curve for the model.
    """
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train)
    train_scores=np.mean(train_scores, axis=1)
    test_scores=np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores, label='Training Accuracy')
    plt.plot(train_sizes, test_scores, label='Test Accuracy')
    plt.legend()
    if zoom_out:
        plt.ylim(0,1.05)

    plt.show()

