import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


class PreparingCSV:
    def __init__(self, input_file, weather_file, output_file, geo_file):
        # Инициализация путей к файлам
        self.input_file = input_file    # CSV с данными рейсов
        self.weather_file = weather_file # CSV с погодными данными
        self.output_file = output_file   # Файл для сохранения результата
        self.geo_file = geo_file         # CSV с координатами аэропортов

    def load_data(self):
        # Загрузка данных из CSV
        print("Загрузка данных...")
        flights_data = pd.read_csv(self.input_file)     # Данные о рейсах
        weather_data = pd.read_csv(self.weather_file)   # Погодные данные
        geo_data = pd.read_csv(self.geo_file)           # Геоданные
        # Сохранение описаний в файлы
        print(flights_data.describe().to_csv("my_description.csv"))
        print(weather_data.describe().to_csv("my_description_weather.csv"))
        return flights_data, weather_data, geo_data

    def preprocess(self, flights_data, weather_data, geo_data, random_state):
        # Основной метод предобработки
        print("Предобработка данных...")

        # Обработка задержек: отрицательные значения -> 0
        flights_data.loc[:, 'Delay_LastAircraft'] = flights_data['Delay_LastAircraft'].apply(lambda x: max(x, 0))

        # Случайная выборка 40% данных для ускорения обработки
        flights_data = flights_data.sample(frac=0.4, random_state=random_state).reset_index(drop=True)

        # Конвертация дат в datetime
        flights_data['FlightDate'] = pd.to_datetime(flights_data['FlightDate'])
        weather_data['FlightDate'] = pd.to_datetime(weather_data['time'])  # Переименование столбца

        # Функция для ограничения значений в столбце
        def clip_values(df, column, lower, upper):
            df[column] = np.clip(df[column], lower, upper)
            return df

        # Обработка выбросов
        flights_data = clip_values(flights_data, 'Arr_Delay', 0, 120)          # Максимальная задержка 2 часа
        flights_data = clip_values(flights_data, 'Delay_LastAircraft', 0, 120)

        # Сортировка по дате рейса
        flights_data = flights_data.sort_values(by='FlightDate').reset_index(drop=True)

        # Объединение данных рейсов с погодой в аэропортах назначения
        merged_data = pd.merge(
            flights_data,
            weather_data,
            how='left',  # Left join
            left_on=['Arr_Airport', 'FlightDate'],  # Ключи для объединения
            right_on=['airport_id', 'FlightDate']
        )
        # Объединение с погодой в аэропортах вылета
        merged_data = pd.merge(
            merged_data,
            weather_data,
            how='left',
            left_on=['Dep_Airport', 'FlightDate'],
            right_on=['airport_id', 'FlightDate'],
            suffixes=('', '_dep')  # Суффиксы для колонок
        )

        # Добавление геоданных аэропортов
        merged_data = pd.merge(
            merged_data,
            geo_data,
            how='left',
            left_on=['Dep_Airport'],
            right_on=['IATA_CODE']  # Код IATA аэропорта
        )

        # Удаление ненужных столбцов
        merged_data.drop(columns=['Dep_Delay_Tag', "Dep_Delay", "STATE",
                                "Dep_CityName", "AIRPORT", "CITY", "Arr_CityName", "Delay_Carrier", "Delay_Weather",
                                  "Delay_NAS", "Delay_Security", "LATITUDE", "LONGITUDE"], inplace=True)

        # Заполнение пропусков медианными значениями
        merged_data.fillna(merged_data.median(numeric_only=True), inplace=True)

        def calculate_flight_order(data):
            # Расчет порядка рейтов самолета за день
            time_order = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}  # Временные метки
            data['DepTime_order'] = data['DepTime_label'].map(time_order)  # Числовое представление времени

            # Сортировка для правильного порядка рейсов
            data = data.sort_values(by=['Tail_Number', 'FlightDate', 'DepTime_order']).reset_index(drop=True)

            # Расчет номера рейса за день для каждого самолета
            data['Flight_Order'] = data.groupby(['Tail_Number', 'FlightDate']).cumcount() + 1

            # Накопительная сумма задержек предыдущих рейсов
            data['PreviousFlights_Delay'] = data.groupby(['Tail_Number', 'FlightDate'])['Delay_LastAircraft'].cumsum() - data['Delay_LastAircraft']
            data['PreviousFlights_Delay'].fillna(0, inplace=True)  # Заполнение NaN для первого рейса

            # Дополнительные признаки
            data['Daily_Departure_Count'] = data.groupby(['FlightDate', 'Dep_Airport'])['Flight_Order'].transform('count')
            data['DayOfWeek'] = data['FlightDate'].dt.dayofweek  # День недели (1-7)
            data['Month'] = data['FlightDate'].dt.month          # Месяц (1-12)
            data['TempDiff'] = data['tmax'] - data['tmin']       # Разница температур
            return data

        # Фильтрация данных только для аэропорта JFK
        merged_data = merged_data[merged_data['Dep_Airport'].isin(['JFK'])]
        merged_data = calculate_flight_order(merged_data)

        # One-Hot Encoding для категориальных признаков
        merged_data = pd.get_dummies(merged_data, columns=['Airline', 'DepTime_label'])

        # Визуализация корреляций
        plt.figure(figsize=(20, 20))
        sns.heatmap(merged_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Тепловая карта корреляций")
        plt.savefig("heatmap.png")
        plt.close()

        return merged_data

    def save_data(self, merged_data):
        # Сохранение обработанных данных
        merged_data.to_csv(self.output_file, index=False)
        print(f"Данные сохранены в {self.output_file}")
        print("Пример данных:")
        print(merged_data.head())
