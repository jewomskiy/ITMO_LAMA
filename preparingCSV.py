import pandas as pd
import numpy as np

input_file = 'data/US_flights_2023.csv'

output_file = 'data/prepared_data.csv'

print("Loading data...")
data = pd.read_csv(input_file)


columns_to_select = ['FlightDate', 'Tail_Number', 'Dep_Airport', 'Arr_Delay', 'Arr_Airport', 'Delay_LastAircraft']
data = data[columns_to_select]

data = data.sample(frac=0.5, random_state=59).reset_index(drop=True)
print(f"Dataset size after sampling: {data.shape}")

data['Arr_Delay'] = np.where(data['Arr_Delay'] > 0, 1, 0)


data.to_csv(output_file, index=False)
print(f"Prepared dataset saved to {output_file}")

print("Sample of prepared data:")
print(data.head())
