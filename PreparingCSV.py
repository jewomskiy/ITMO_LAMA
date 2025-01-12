
import pandas as pd
import numpy as np
from datetime import datetime

class PreparingCSV:
    def __init__(self, input_file, weather_file, output_file):
        self.input_file = input_file
        self.weather_file = weather_file
        self.output_file = output_file

    def load_data(self):
        print("Loading data...")
        flights_data = pd.read_csv(self.input_file)
        weather_data = pd.read_csv(self.weather_file)
        return flights_data, weather_data

    def preprocess(self, flights_data, weather_data, random_state):
        print("Preprocessing data...")

        # Selecting required columns
        flights_columns = ['FlightDate', 'Tail_Number', 'Dep_Airport', 'Arr_Delay', 'Arr_Airport', 'Delay_LastAircraft']
        flights_data = flights_data[flights_columns].copy()

        # Setting negative delays to 0
        flights_data.loc[:, 'Arr_Delay'] = flights_data['Arr_Delay'].apply(lambda x: max(x, 0))
        upper_bound = flights_data['Arr_Delay'].quantile(0.95)
        flights_data['Arr_Delay'] = flights_data['Arr_Delay'].clip(upper=upper_bound)

        # Sampling 50% of the data
        flights_data = flights_data.sample(frac=0.1, random_state=random_state).reset_index(drop=True)

        # Formatting dates
        flights_data['FlightDate'] = pd.to_datetime(flights_data['FlightDate'])
        weather_data['FlightDate'] = pd.to_datetime(weather_data['time'])

        # Reducing rows with no delay (Arr_Delay == 0) by 50%
        no_delay = flights_data[flights_data['Arr_Delay'] == 0]
        delayed = flights_data[flights_data['Arr_Delay'] != 0]
        if len(no_delay) > 1:  # Ensure sufficient data for sampling
            no_delay_sample = no_delay.sample(frac=0.5, random_state=random_state)
            flights_data = pd.concat([no_delay_sample, delayed]).reset_index(drop=True)

        # Sorting by FlightDate
        flights_data = flights_data.sort_values(by='FlightDate').reset_index(drop=True)

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
        merged_data = merged_data.dropna(thresh=len(merged_data.columns) * 0.8)  # Удаление строк, где пропущено >20%
        merged_data.fillna(merged_data.median(numeric_only=True), inplace=True)

        print(f"Prepared dataset size: {merged_data.shape}")
        return merged_data

    def save_data(self, merged_data):
        merged_data.to_csv(self.output_file, index=False)
        print(f"Data saved to {self.output_file}")
        print("Sample of prepared data:")
        print(merged_data.head())
