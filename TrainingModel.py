# Импорт библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task

class TrainingModel:
    def __init__(self, data_path, target_name, n_threads, timeout, n_folds):
        # Параметры обучения
        self.data_path = data_path    # Путь к данным
        self.target_name = target_name # Название целевой переменной
        self.n_threads = n_threads    # Число потоков
        self.timeout = timeout        # Время обучения (сек)
        self.n_folds = n_folds        # Число фолдов

    def load_and_split_data(self, random_state):
        # Загрузка данных и удаление строк с NaN
        data = pd.read_csv(self.data_path)
        data = data.dropna()  # Удаление пропущенных значений

        # Разделение на тренировочную и тестовую выборки
        train_data, test_data = train_test_split(
            data,
            test_size=0.3,           # 30% тестовых данных
            random_state=random_state, # Фиксация случайного состояния
            shuffle=True              # Перемешивание данных
        )
        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

    def train_model(self, train_data, test_data, random_state):
        # Настройка задачи регрессии
        task = Task('reg')  # Регрессия

        # Параметры для AutoML
        roles = {'target': self.target_name}  # Указание целевой переменной

        # Инициализация AutoML
        automl_model = TabularUtilizedAutoML(
            task=task,
            timeout=self.timeout,     # Макс. время обучения
            cpu_limit=self.n_threads, # Лимит потоков CPU
            reader_params={
                'n_jobs': self.n_threads, # Потоки для чтения данных
                'cv': self.n_folds,        # Кросс-валидация
                'random_state': random_state  # Семя для воспроизводимости
            },
        )

        # Обучение модели с кросс-валидацией
        oof_predictions = automl_model.fit_predict(train_data, roles=roles, verbose=2)
        # Предсказание на тестовых данных
        test_predictions = automl_model.predict(test_data)
        return automl_model, oof_predictions, test_predictions