from PreparingCSV import PreparingCSV
from TrainingModel import TrainingModel
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import gdown
url = 'https://drive.google.com/drive/folders/18houVS5ebR_Bw3_lQ3bNZ5X09lhaFsim'

def main():
    # Step 1: Data Preparation
    TARGET_NAME = 'Arr_Delay'
    RANDOM_STATE = 41
    N_THREADS = 6
    timeout = 60 * 60 * 8
    np.random.seed(RANDOM_STATE)
    torch.set_num_threads(N_THREADS)
    gdown.download_folder(url)
    preparator = PreparingCSV(
        input_file='data/US_flights_2023.csv',
        weather_file='data/weather_meteo_by_airport.csv',
        output_file='data/prepared_data.csv'
    )
    flights, weather = preparator.load_data()
    prepared_data = preparator.preprocess(flights, weather)
    preparator.save_data(prepared_data)

    # Step 2: Model Training
    model_trainer = TrainingModel(
        data_path='data/prepared_data.csv',
        target_name='Arr_Delay',
        n_threads=N_THREADS,
        timeout=timeout,
        n_folds=5
    )
    train_data, test_data = model_trainer.load_and_split_data()
    automl_model, oof_predictions, test_predictions = model_trainer.train_model(train_data, test_data)

    # Check data before output our test our model
    if np.any(np.isnan(oof_predictions.data)) or np.any(np.isnan(train_data[TARGET_NAME].values)):
        print("Detected NaN in predictions or target values.")
        print(f"NaN in oof_preds: {np.isnan(oof_predictions.data).sum()}")
        print(f"NaN in target: {np.isnan(train_data[TARGET_NAME].values).sum()}")
    else:
        print(f'OOF score: {roc_auc_score(train_data[TARGET_NAME].values, oof_predictions.data[:, 0])}')


    if np.any(np.isnan(test_predictions.data)) or np.any(np.isnan(test_data[TARGET_NAME].values)):
        print("Detected NaN in test predictions or test target values.")
        print(f"NaN in te_preds: {np.isnan(test_predictions.data).sum()}")
        print(f"NaN in test target: {np.isnan(test_data[TARGET_NAME].values).sum()}")
    else:
        print(f'HOLDOUT score: {roc_auc_score(test_data[TARGET_NAME].values, test_predictions.data[:, 0])}')
    print("Pipeline execution completed.")

if __name__ == "__main__":
    main()
