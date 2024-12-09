# Essential DS libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task


class TrainingModel:
    def __init__(self, data_path, target_name, n_threads, timeout, n_folds):
        self.data_path = data_path
        self.target_name = target_name
        self.n_threads = n_threads
        self.timeout = timeout
        self.n_folds = n_folds

    def load_and_split_data(self):
        data = pd.read_csv(self.data_path)
        train_data, test_data = train_test_split(
            data,
            test_size=0.1,
            stratify=data[self.target_name],
            random_state=59
        )
        return train_data.dropna(), test_data.dropna()

    def train_model(self, train_data, test_data):
        task = Task('binary', loss='logloss', metric='auc')
        roles = {
            'target': self.target_name,
            'drop': ['FlightDate', 'Dep_Airport', 'Arr_Airport', 'airport_id_dep', 'time_dep']
        }

        automl_model = TabularUtilizedAutoML(
            task=task,
            timeout=self.timeout,
            cpu_limit=self.n_threads,
            reader_params={'n_jobs': self.n_threads, 'cv': self.n_folds, 'random_state': 59}
        )

        oof_predictions = automl_model.fit_predict(train_data, roles=roles, verbose=1)
        print("Model training completed.")
        test_predictions = automl_model.predict(test_data)
        return automl_model, oof_predictions, test_predictions