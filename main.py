import sys
from PreparingCSV import PreparingCSV
from TrainingModel import TrainingModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_log_error
import numpy as np
import torch
import gdown
from pathlib import Path
import matplotlib.pyplot as plt
import shap

# Ссылка на папку Google Drive с данными
url = 'https://drive.google.com/drive/folders/18houVS5ebR_Bw3_lQ3bNZ5X09lhaFsim'

def main():
    # Параметры конфигурации
    TARGET_NAME = 'Arr_Delay'  # Целевая переменная - задержка прибытия
    RANDOM_STATE = int(input('Введите значение random state: '))  # Переменная для воспроизводимости
    N_THREADS = int(input('Введите число потоков: '))  # Количество CPU-потоков
    MIN = int(input('Введите время работы (в минутах): '))  # Время обучения модели
    timeout = 60 * MIN  # Конвертация в секунды

    # Перенаправление вывода в лог-файл
    with open('output.log', 'w') as log_file:
        sys.stdout = log_file
        try:
            # Инициализация случайных состояний для воспроизводимости
            np.random.seed(RANDOM_STATE)
            torch.set_num_threads(N_THREADS)  # Ограничение потоков для PyTorch

            # Проверка наличия данных и загрузка при необходимости
            flights_data = Path("data/US_flights_2023.csv")
            weather_data = Path("data/weather_meteo_by_airport.csv")
            if not flights_data.exists() or not weather_data.exists():
                gdown.download_folder(url)  # Скачивание данных через gdown

            # Шаг 1: Подготовка данных
            preparator = PreparingCSV(
                input_file='data/US_flights_2023.csv',  # Исходные данные рейсов
                weather_file='data/weather_meteo_by_airport.csv',  # Погодные данные
                output_file='data/prepared_data.csv',  # Выходной файл
                geo_file='data/airports_geolocation.csv'  # Геоданные аэропортов
            )
            # Загрузка и предобработка
            flights, weather, geo = preparator.load_data()
            prepared_data = preparator.preprocess(flights, weather, geo, RANDOM_STATE)
            preparator.save_data(prepared_data)

            # Шаг 2: Обучение модели
            model_trainer = TrainingModel(
                data_path='data/prepared_data.csv',  # Путь к обработанным данным
                target_name=TARGET_NAME,  # Целевая переменная
                n_threads=N_THREADS,  # Потоки для вычислений
                timeout=timeout,  # Лимит времени обучения
                n_folds=8  # Количество фолдов кросс-валидации
            )
            # Разделение данных и обучение
            train_data, test_data = model_trainer.load_and_split_data(RANDOM_STATE)
            automl_model, oof_predictions, test_predictions = model_trainer.train_model(train_data, test_data, RANDOM_STATE)

            # Проверка на NaN перед вычислением метрик
            if np.any(np.isnan(test_predictions.data)) or np.any(np.isnan(test_data[TARGET_NAME].values)):
                print("Обнаружены NaN в предсказаниях или тестовых данных.")
            else:
                # Вычисление стандартных метрик регрессии
                mse = mean_squared_error(test_data[TARGET_NAME].values, test_predictions.data)
                mae = mean_absolute_error(test_data[TARGET_NAME].values, test_predictions.data)
                r2 = r2_score(test_data[TARGET_NAME].values, test_predictions.data)
                print(f'MSE: {mse}', f'MAE: {mae}', f'R2: {r2}', sep='\n')

                # Обработка отрицательных предсказаний для RMSLE
                oof_predictions.data[:, 0] = np.clip(oof_predictions.data[:, 0], 0, None)
                test_predictions.data[:, 0] = np.clip(test_predictions.data[:, 0], 0, None)

                # Вычисление RMSLE с защитой от NaN
                oof_pred_exp = np.nan_to_num(np.exp(oof_predictions.data[:, 0]) - 1, nan=0.0)
                test_pred_exp = np.nan_to_num(np.exp(test_predictions.data[:, 0]) - 1, nan=0.0)
                oof_score = root_mean_squared_log_error(train_data[TARGET_NAME].values, oof_pred_exp)
                holdout_score = root_mean_squared_log_error(test_data[TARGET_NAME].values, test_pred_exp)
                print(f"OOF RMSLE: {oof_score}", f"HOLDOUT RMSLE: {holdout_score}", sep='\n')

                # Сохранение предсказаний
                results_df = test_data[['FlightDate', 'Dep_Airport', 'Arr_Airport', 'Tail_Number', 'Flight_Order', TARGET_NAME]].copy()
                results_df['Predicted_Arr_Delay'] = test_predictions.data[:, 0].round(1)  # Округление до 1 знака
                results_df.to_csv('ModelAnswer.csv', index=False)  # Сохранение в CSV

            print("Пайплайн успешно выполнен.")
        finally:
            sys.stdout = sys.__stdout__  # Восстановление стандартного вывода

if __name__ == "__main__":
    main()