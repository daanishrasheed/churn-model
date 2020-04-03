import os

# Directory paths
SRC_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'data')
RAW_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'raw')
INTERIM_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'interim')
PROCESSED_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'processed')

# File paths - Raw data
RAW_DATA_PATH = os.path.join(RAW_DATA_DIRECTORY, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
