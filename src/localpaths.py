import os

# Directory paths
SRC_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_FEATURES_DIRECTORY = os.path.join(SRC_DIRECTORY, 'features')
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')
RAW_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'raw')
INTERIM_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'interim')
PROCESSED_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'processed')

# File paths - Raw data
RAW_DATA_PATH = os.path.join(RAW_DATA_DIRECTORY, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Raw train-test-split data
X_TRAIN_RAW_PATH = os.path.join(RAW_DATA_DIRECTORY, 'X_train.csv')
X_TEST_RAW_PATH = os.path.join(RAW_DATA_DIRECTORY, 'X_test.csv')
Y_TRAIN_RAW_PATH = os.path.join(RAW_DATA_DIRECTORY, 'y_train.csv')
Y_TEST_RAW_PATH = os.path.join(RAW_DATA_DIRECTORY, 'y_test.csv')

# File paths - Clean train-test split data
X_TRAIN_CLEAN_PATH = os.path.join(INTERIM_DATA_DIRECTORY, 'X_train.csv')
Y_TRAIN_CLEAN_PATH = os.path.join(INTERIM_DATA_DIRECTORY, 'y_train.csv')

# File path - Featurized train-test split data ready for modeling
X_TRAIN_FEATURIZED_PATH = os.path.join(PROCESSED_DATA_DIRECTORY, 'X_train.csv')
Y_TRAIN_FEATURIZED_PATH = os.path.join(PROCESSED_DATA_DIRECTORY, 'y_train.csv')