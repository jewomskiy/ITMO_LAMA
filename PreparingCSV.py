
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
        print(flights_data.describe().to_csv("my_description.csv"))
        print(weather_data.describe().to_csv("my_description_weather.csv"))
        return flights_data, weather_data

    def preprocess(self, flights_data, weather_data, random_state):
        print("Preprocessing data...")

        # Selecting required columns
        flights_columns = ['FlightDate', 'Day_Of_Week', 'Tail_Number', 'Dep_Airport', 'Arr_Delay', 'Arr_Airport', 'Delay_LastAircraft',
                           'Aicraft_age', 'Dep_Delay']
        flights_data = flights_data[flights_columns].copy()

        # Setting negative delays to 0
        flights_data.loc[:, 'Delay_LastAircraft'] = flights_data['Delay_LastAircraft'].apply(lambda x: max(x, 1))

        # Sampling 50% of the data
        flights_data = flights_data.sample(frac=0.35, random_state=random_state).reset_index(drop=True)

        # Formatting dates
        flights_data['FlightDate'] = pd.to_datetime(flights_data['FlightDate'])
        weather_data['FlightDate'] = pd.to_datetime(weather_data['time'])

        # - Для категориальных колонок можно заполнить наиболее частым значением
        def clip_values(df, column, lower, upper):
            df[column] = np.clip(df[column], lower, upper)
            return df

        # Устанавливаем разумные границы для каждого показателя
        flights_data = clip_values(flights_data, 'Arr_Delay', 0, 60)
        flights_data = clip_values(flights_data, 'Delay_LastAircraft', 0, 45)
        flights_data = clip_values(flights_data, 'Dep_Delay', 0, 60)
        weather_data = clip_values(weather_data, 'tavg', -40, 40)  # Средняя температура
        weather_data = clip_values(weather_data, 'tmin', -50, 35)  # Минимальная температура
        weather_data = clip_values(weather_data, 'wspd', 0, 50)  # Скорость ветра (макс. ~50 м/с)
        weather_data = clip_values(weather_data, 'pres', 970, 1050)  # Давление (средние значения на уровне моря)
        weather_data = clip_values(weather_data, 'snow', 0, 1000) #Снег
        weather_data = clip_values(weather_data, 'prcp', 0, 300) #осадки

        # 4. Проверка финального набора данных
        print("Обработанные данные о погоде:")
        print(weather_data.describe())

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
