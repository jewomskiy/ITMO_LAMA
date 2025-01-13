# Essential DS libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML, TabularAutoML
from lightautoml.tasks import Task


class TrainingModel:
    def __init__(self, data_path, target_name, n_threads, timeout, n_folds):
        self.data_path = data_path
        self.target_name = target_name
        self.n_threads = n_threads
        self.timeout = timeout
        self.n_folds = n_folds

    def load_and_split_data(self, random_state):
        data = pd.read_csv(self.data_path)
        data = data.dropna()
        print(f"Loaded dataset size: {data.shape}")

        train_data, test_data = train_test_split(
            data,
            test_size=0.2,
            random_state=random_state
        )
        return train_data, test_data

    def train_model(self, train_data, test_data, random_state):
        task = Task('reg')
        roles = {
            'target': self.target_name,
            'drop': ['Dep_Airport', 'Arr_Airport', 'airport_id_dep', 'time_dep', 'Tail_Number', 'time']
        }

        automl_model = TabularUtilizedAutoML(
            task=task,
            timeout=self.timeout,
            cpu_limit=self.n_threads,
            reader_params={'n_jobs': self.n_threads, 'cv': self.n_folds, 'random_state': random_state},
            configs_list=['config_path.yml']
        )



        oof_predictions = automl_model.fit_predict(train_data, roles=roles, verbose=2)
        print("Model training completed.")
        test_predictions = automl_model.predict(test_data)
        return automl_model, oof_predictions, test_predictions
