import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
        flights_columns = ['FlightDate', 'Tail_Number', 'Dep_Delay', 'Airline', 'Dep_Airport', 'Arr_Airport', 'Delay_LastAircraft', 'Flight_Duration', 'DepTime_label']
        flights_data = flights_data[flights_columns].copy()

        # Setting negative delays to 0
        flights_data.loc[:, 'Delay_LastAircraft'] = flights_data['Delay_LastAircraft'].apply(lambda x: max(x, 0))

        # Sampling 50% of the data
        flights_data = flights_data.sample(frac=0.6, random_state=random_state).reset_index(drop=True)

        # Formatting dates
        flights_data['FlightDate'] = pd.to_datetime(flights_data['FlightDate'])
        weather_columns = ['tmin', 'tmax', 'time', 'airport_id']
        weather_data = weather_data[weather_columns].copy()
        weather_data['FlightDate'] = pd.to_datetime(weather_data['time'])

        # - Для категориальных колонок можно заполнить наиболее частым значением
        def clip_values(df, column, lower, upper):
            df[column] = np.clip(df[column], lower, upper)
            return df

        flights_data = clip_values(flights_data, 'Dep_Delay', 0, 60)
        flights_data = clip_values(flights_data, 'Delay_LastAircraft', 0, 60)
        # weather_data = clip_values(weather_data, 'tavg', -40, 40)  # Средняя температура
        # weather_data = clip_values(weather_data, 'tmin', -50, 35)  # Минимальная температура
        # weather_data = clip_values(weather_data, 'wspd', 0, 50)  # Скорость ветра (макс. ~50 м/с)
        # weather_data = clip_values(weather_data, 'prcp', 0, 300) #осадки

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
        # merged_data = merged_data.dropna(thresh=len(merged_data.columns) * 0.8)  # Удаление строк, где пропущено >20%
        merged_data.fillna(merged_data.median(numeric_only=True), inplace=True)

        def calculate_flight_order(data):
            # Создаём вспомогательный столбец для сортировки по времени вылета
            time_order = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
            data['DepTime_order'] = data['DepTime_label'].map(time_order)

            # Сортируем данные для корректного вычисления порядка
            data = data.sort_values(by=['Tail_Number', 'FlightDate', 'DepTime_order']).reset_index(drop=True)

            # Группировка по Tail_Number и FlightDate
            groups = data.groupby(['Tail_Number', 'FlightDate'])

            # Добавляем порядковый номер вылета
            data['Flight_Order'] = groups.cumcount() + 1

            # Проверка совпадения аэропорта вылета и прилёта предыдущего рейса
            data['Valid_Dep_Airport'] = groups['Arr_Airport'].shift() == data['Dep_Airport']

            # Рассчитываем накопительную сумму задержек за день
            data['PreviousFlights_Delay'] = groups['Delay_LastAircraft'].cumsum() - data['Delay_LastAircraft']

            # Убираем вспомогательный столбец
            data.drop(columns=['DepTime_order'], inplace=True)
            data.drop(columns=['Valid_Dep_Airport'], inplace=True)

            # Заполняем значения NaN в первых рейсах дня
            data['PreviousFlights_Delay'].fillna(0, inplace=True)

            # Добавляем новый столбец для подсчета количества вылетов за день для выбранного аэропорта
            data['Daily_Departure_Count'] = data.groupby(['FlightDate', 'Dep_Airport'])['Flight_Order'].transform(
                'count')

            data['DayOfWeek'] = data['FlightDate'].dt.dayofweek

            # Добавляем признак "месяц"
            data['Month'] = data['FlightDate'].dt.month

            # Добавляем признак "разница температур"
            data['TempDiff'] = data['tmax'] - data['tmin']

            return data

        def add_avg_delay_percentage(data, window=14):
            # Создаем колонку с бинарным индикатором задержки (1 - задержка, 0 - нет)
            data['Is_Delayed'] = (data['Dep_Delay'] > 0).astype(int)

            # Группируем по аэропорту и дате, чтобы рассчитать процент задержек за день
            daily_delay_percentage = data.groupby(['Dep_Airport', 'FlightDate'])['Is_Delayed'].mean().reset_index()
            daily_delay_percentage.rename(columns={'Is_Delayed': 'Daily_Delay_Percentage'}, inplace=True)

            # Добавляем колонку с датой для расчета скользящего среднего
            daily_delay_percentage['FlightDate'] = pd.to_datetime(daily_delay_percentage['FlightDate'])

            # Сортируем данные для корректного расчета скользящего среднего
            daily_delay_percentage = daily_delay_percentage.sort_values(by=['Dep_Airport', 'FlightDate'])

            # Рассчитываем скользящее среднее за предыдущие N дней
            daily_delay_percentage['Avg_Delay_Percentage_Last_N_Days'] = daily_delay_percentage.groupby('Dep_Airport')[
                'Daily_Delay_Percentage'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )

            # Объединяем с основным датасетом
            data = pd.merge(data,
                            daily_delay_percentage[['Dep_Airport', 'FlightDate', 'Avg_Delay_Percentage_Last_N_Days']],
                            on=['Dep_Airport', 'FlightDate'], how='left')

            return data

        # Применяем функцию
        merged_data = calculate_flight_order(merged_data)
        merged_data = add_avg_delay_percentage(merged_data)
        merged_data = merged_data[merged_data['Dep_Airport'].isin(['JFK'])]
        # winter_months = [6, 7, 8]
        # merged_data = merged_data[merged_data['FlightDate'].dt.month.isin(winter_months)]
        # merged_data = pd.get_dummies(merged_data, columns=['Airline'], prefix='Airline')
        merged_data = pd.get_dummies(merged_data, columns=['DepTime_label'])
        print(f"Prepared dataset size: {merged_data.shape}")

        merged_data = merged_data.sort_values(by='FlightDate').reset_index(drop=True)

        # Save heatmap of correlations
        plt.figure(figsize=(20, 20))
        sns.heatmap(merged_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("heatmap.png")
        plt.close()

        return merged_data

    def save_data(self, merged_data):
        merged_data.to_csv(self.output_file, index=False)
        print(f"Data saved to {self.output_file}")
        print("Sample of prepared data:")
        print(merged_data.head())
