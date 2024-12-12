import sys
from PreparingCSV import PreparingCSV
from TrainingModel import TrainingModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import torch
import gdown
from pathlib import Path

url = 'https://drive.google.com/drive/folders/18houVS5ebR_Bw3_lQ3bNZ5X09lhaFsim'

def main():
    TARGET_NAME = 'Arr_Delay'
    RANDOM_STATE = int(input('Enter a random state value: '))
    N_THREADS = int(input('Enter number of threads: '))
    HOURS = int(input('How many hours do you want? '))
    timeout = 60 * HOURS
    # Перенаправляем вывод в файл
    with open('output.log', 'w') as log_file:
        sys.stdout = log_file  # Перенаправление stdout в файл
        try:

            np.random.seed(RANDOM_STATE)
            torch.set_num_threads(N_THREADS)
            flights_data = Path("data/US_flights_2023.csv")
            weather_data = Path("data/weather_meteo_by_airport.csv")
            if not flights_data.exists() or not weather_data.exists():
                gdown.download_folder(url)
            preparator = PreparingCSV(
                input_file='data/US_flights_2023.csv',
                weather_file='data/weather_meteo_by_airport.csv',
                output_file='data/prepared_data.csv'
            )
            flights, weather = preparator.load_data()
            prepared_data = preparator.preprocess(flights, weather, RANDOM_STATE)
            preparator.save_data(prepared_data)

            # Step 2: Model Training
            model_trainer = TrainingModel(
                data_path='data/prepared_data.csv',
                target_name='Arr_Delay',
                n_threads=N_THREADS,
                timeout=timeout,
                n_folds=5
            )
            train_data, test_data = model_trainer.load_and_split_data(RANDOM_STATE)
            automl_model, oof_predictions, test_predictions = model_trainer.train_model(train_data, test_data)

            # Check data before output our test our model
            if np.any(np.isnan(test_predictions.data)) or np.any(np.isnan(test_data[TARGET_NAME].values)):
                print("Detected NaN in test predictions or test target values.")
                print(f"NaN in test_preds: {np.isnan(test_predictions.data).sum()}")
                print(f"NaN in test target: {np.isnan(test_data[TARGET_NAME].values).sum()}")
            else:
                # Вычисляем метрики для задачи регрессии
                mse = mean_squared_error(test_data[TARGET_NAME].values, test_predictions.data)
                mae = mean_absolute_error(test_data[TARGET_NAME].values, test_predictions.data)
                r2 = r2_score(test_data[TARGET_NAME].values, test_predictions.data)

                print(f'Mean Squared Error: {mse}')
                print(f'Mean Absolute Error: {mae}')
                print(f'R-squared: {r2}')
            print("Pipeline execution completed.")
        finally:
            sys.stdout = sys.__stdout__  # Восстановление стандартного вывода

if __name__ == "__main__":
    main()
