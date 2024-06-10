import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, random_split

#clean the rates dataset 
print('Loading ../../data/original/rates.xlsx')
df = pd.read_excel('../../data/original/rates.xlsx')
print('Loaded ../../data/original/rates.xlsx')

# the column titles are improperly aligned
df = df.rename(columns={
    'Valley Fever Cases and Incidence Rates by Local Health Jurisdiction, California, 2001-2022 ' : 'county',
    'Unnamed: 1' : 'year', 
    'Unnamed: 2' : 'cases', 
    'Unnamed: 3' : 'rates'
},)

# remove rates because the sight said thate the rate is unreliable
df.drop('rates', axis=1, inplace=True)

# drop first and last row because they are not instances of the data
df.drop(index=[df.index[0], df.index[-1]], axis=0, inplace=True)
df['county'] = df['county'].str.lower()

#save the newly created cases dataset for later
cases = df.copy()

#start clearning the weather dataset
print('Loading weather.csv')
df = pd.read_csv('../../data/original/weather.csv')
print('Loaded weather.csv')
# drop all unnessarcary columns
cols = ['dt', 'timezone', 'lat', 'lon', 'weather_icon', 'sea_level', 'grnd_level']
df.drop(cols, axis=1, inplace=True)
# fill all missing valuues
df = df.fillna(0)

#convert to a datetime object
df['dt_iso'] = pd.to_datetime(df['dt_iso'], format='%Y-%m-%d %H:%M:%S %z UTC').dt.strftime('%Y-%m-%d %H')
df['dt_iso'] = pd.to_datetime(df['dt_iso'])

#fix nameing for merge
df = df.rename(columns={'city_name':'county'})

# cartedoricaly encode the weather main and description, was going to one hot it but it caused an explosion of feature that would make our models over fit
encoder = LabelEncoder()
df['weather_main'] = encoder.fit_transform(df['weather_main'])
df['weather_description'] = encoder.fit_transform(df['weather_description'])

#aggregate to daily averages
df = df.set_index('dt_iso')
daily_averages = df.groupby('county').resample('D').mean()
df = daily_averages.copy()
df = df.reset_index(level=['county', 'dt_iso'])
df = df.rename(columns={'dt_iso': 'date'})

#fix more naming conventions for merge
relabelled_df = df.copy()
relabelled_df['county'] = df['county'].str.replace(' County', '')
relabelled_df['county'] = relabelled_df['county'].str.lower()
relabelled_df['county'] = relabelled_df['county'].str.replace(' valley', '')

#split dateimet so that there is a year feature to nmerge the two dfs on
df = relabelled_df.copy()
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
# save weather as the weather df
weather = df.copy()
#merge the two datasets so that they match the number of cases to each year and county
merged_df = pd.merge(weather, cases, on=['county','year'],how='inner')
print(f'Merged DF shape: {merged_df.shape}')

#cutoff date fo the data so that it can early forecast the cases is august 31
day_threshold = 31
month_threshold = 8

# Create a boolean mask
mask = (merged_df['date'].dt.day <= day_threshold) & (merged_df['date'].dt.month <= month_threshold)

# Filter the DataFrame
filtered_merged_df = merged_df.loc[mask]


df = filtered_merged_df.copy()


#use min max normalization on all the data
cols_to_normalize = [
    'temp',  
    'visibility', 
    'dew_point', 
    'feels_like',
    'temp_min', 
    'temp_max', 
    'pressure', 
    'humidity', 
    'wind_speed',
    'wind_deg', 
    'wind_gust', 
    'rain_1h', 
    'rain_3h', 
    'snow_1h', 
    'snow_3h',
    'clouds_all', 
    'weather_id', 
    'weather_main', 
    'weather_description'
    ]

scaler = MinMaxScaler()

df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

#create the sequences for the lstm so that they are all the same size
grouped = df.groupby(['county', 'year'])
x_list = []
y_list = []

for _, group_df in grouped:
    x = group_df.drop(['county', 'year', 'cases', 'month', 'date', 'day'], axis=1)
    x = x.iloc[:243]
    x = x.values 
    x_list.append(x)

    y = group_df['cases'].values[-1] 
    y_list.append(y)

x = np.stack(x_list) 
y = np.array(y_list)

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

print(f'X tensor shape: {x_tensor.shape}, Y tensor shape: {y_tensor.shape}')

## add augmented noise to 4x the size of the dataset 
aug = []
for i in range(4):
    noise = torch.randn_like(x_tensor) * 0.001
    aug.append(x_tensor + noise)

aug.append(x_tensor)
augmented_x = torch.cat(aug, dim=0)

augmented_y = y_tensor.repeat(5)

print(augmented_x.shape)  
print(augmented_y.shape)  
dataset = TensorDataset(augmented_x, augmented_y)

total_size = len(dataset)
train_size = int(total_size * 0.8)
val_size = int(total_size * 0.15)
test_size = total_size - train_size - val_size

#split the datasets into 80 15 and 5% for train val and test
train, val, test = random_split(dataset, [train_size, val_size, test_size])

print(f'Train Size: {len(train)}, Val Size: {len(val)}, Test Size: {len(test)}')
torch.save(train, '../../data/cleaned/train.pt')
torch.save(val, '../../data/cleaned/val.pt')
torch.save(test, '../../data/cleaned/test.pt')

print('Train, val, and test datasets saved')