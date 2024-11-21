# Standard python libraries
import os
import time
import requests

# Essential DS libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task

RANDOM_STATE = 59
N_THREADS = 6

np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)


TARGET_NAME = 'Arr_Delay'

data = pd.read_csv('data/prepared_data.csv')
data.sample(5)

tr_data, te_data = train_test_split(
    data,
    test_size=0.1,
    stratify=data[TARGET_NAME],
    random_state=RANDOM_STATE
)

tr_data = tr_data.dropna()
te_data = te_data.dropna()

print(f'Data splitted. Parts sizes: tr_data = {tr_data.shape}, te_data = {te_data.shape}')

tr_data.head()

# specify task type
#  'binary' - for binary classification.
#  'reg' - for regression.
#  'multiclass' - for multiclass classification.
task = Task(
    'reg',  # required
    loss='mse',
    metric='mse'
)

# specify feature roles
roles = {
    'target': TARGET_NAME,  # required
    'drop': ['FlightDate', 'Dep_Airport', 'Arr_Airport']
}

N_FOLDS = 5
TIMEOUT = 60 * 45  # 10 minutes

utilized_automl = TabularUtilizedAutoML(
    task=task,
    timeout=TIMEOUT,
    cpu_limit=N_THREADS,
    reader_params={'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
)

tr_data = tr_data.dropna()
te_data = te_data.dropna()

oof_preds = utilized_automl.fit_predict(tr_data, roles=roles, verbose=1)

print(utilized_automl.create_model_str_desc())

te_preds = utilized_automl.predict(te_data)
print(f'Prediction for te_data:\n{te_preds}\nShape = {te_preds.shape}')


print(f'OOF score: {roc_auc_score(tr_data[TARGET_NAME].values, oof_preds.data[:, 0])}')
print(f'HOLDOUT score: {roc_auc_score(te_data[TARGET_NAME].values, te_preds.data[:, 0])}')