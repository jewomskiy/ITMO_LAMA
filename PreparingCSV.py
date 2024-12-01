import pandas as pd
import numpy as np
from datetime import datetime


class DataPreparation:
    def __init__(self, input_file, weather_file, output_file):
        self.input_file = input_file
        self.weather_file = weather_file
        self.output_file = output_file

    def load_data(self):
        print("Loading data...")
        flights_data = pd.read_csv(self.input_file)
        weather_data = pd.read_csv(self.weather_file)
        return flights_data, weather_data

    def preprocess(self, flights_data, weather_data):
        print("Preprocessing data...")

        # Selecting required columns
        flights_columns = ['FlightDate', 'Tail_Number', 'Dep_Airport', 'Arr_Delay', 'Arr_Airport', 'Delay_LastAircraft']
        flights_data = flights_data[flights_columns]

        # Sampling 50% of the data
        flights_data = flights_data.sample(frac=0.5, random_state=59).reset_index(drop=True)

        # Converting `Arr_Delay` to binary
        flights_data['Arr_Delay'] = np.where(flights_data['Arr_Delay'] > 0, 1, 0)

        # Formatting dates
        flights_data['FlightDate'] = pd.to_datetime(flights_data['FlightDate'])
        weather_data['FlightDate'] = pd.to_datetime(weather_data['time'])

        # Merging flights with weather data
        merged_data = pd.merge(
            flights_data,
            weather_data,
            how='left',
            left_on=['Arr_Airport', 'FlightDate'],
            right_on=['airport_id', 'FlightDate']
        )
        merged_data = pd.merge(
            merged_data,
            weather_data,
            how='left',
            left_on=['Dep_Airport', 'FlightDate'],
            right_on=['airport_id', 'FlightDate'],
            suffixes=('', '_dep')
        )

        # Dropping duplicate column
        merged_data.drop(columns=['airport_id'], inplace=True)

        print(f"Prepared dataset size: {merged_data.shape}")
        return merged_data

    def save_data(self, merged_data):
        merged_data.to_csv(self.output_file, index=False)
        print(f"Data saved to {self.output_file}")
        print("Sample of prepared data:")
        print(merged_data.head())
