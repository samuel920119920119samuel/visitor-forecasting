
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

# Data wrangling brought to you by the1owl
# https://www.kaggle.com/the1owl/surprise-me

import os
path = "../data/Weather/1-1-16_5-31-17_Weather" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称

data = {

}

for i in files:
  name = str(i).split('.')[0]
  name = str(name).split(' ')[0]
  data[name] = pd.read_csv('../data/Weather/1-1-16_5-31-17_Weather/' + i)
  data[name] = data[name].fillna(-1)

data['storeVSstation'] = pd.read_csv('../data/Weather/air_store_info_with_nearest_active_station.csv')

data['storeVSstation'] = data['storeVSstation'][['air_store_id', 'station_id']]

import pandas as pd

result = pd.date_range('2016/1/1', periods=517, freq='D')

result = result.strftime('%Y-%m-%d')

unique_stores = data['storeVSstation']['air_store_id'].unique()
stores = pd.concat(
    [
        pd.DataFrame({
            'air_store_id': unique_stores,
            'calendar_date': [result[i]] * len(unique_stores)
        }) for i in range(len(result))
    ],
    axis=0,
    ignore_index=True).reset_index(drop=True) #原index不插入變成一column

data['storeVSstation'].head()

stores = pd.merge(stores, data['storeVSstation'], how='left', on=['air_store_id']) # stores.columns = ['air_store_id', 'cal_date', 'station_id']

stores = stores.sort_values(by = ['air_store_id']).reset_index(drop=True)


stations = pd.DataFrame()
for i in files:
  name = str(i).split('.')[0]
  name = str(name).split(' ')[0]
  data[name]['station_id'] = [name]*len(data[name])
  stations = pd.concat([stations, data[name]], axis=0, ignore_index=True).reset_index(drop=True)

stations.head()

storesVSstations = pd.merge(stores, stations, how='left', on=['station_id', 'calendar_date']) 
# storesVSstations.columns = ['air_store_id', 'cal_date', 'station_id', 'air_store_id', 'calendar_date', 'station_id', 'avg_temperature',
#        'high_temperature', 'low_temperature', 'precipitation',
#        'hours_sunlight', 'solar_radiation', 'deepest_snowfall',
#        'total_snowfall', 'avg_wind_speed', 'avg_vapor_pressure',
#        'avg_local_pressure', 'avg_humidity', 'avg_sea_pressure',
#        'cloud_cover']



storesVSstations[['air_store_id', 'calendar_date', 'avg_humidity', 'avg_temperature', 'precipitation', 'hours_sunlight', 'solar_radiation', 'total_snowfall']].head()

storesVSstations = storesVSstations[['air_store_id', 'calendar_date', 'avg_humidity', 'avg_temperature', 'precipitation', 'hours_sunlight', 'solar_radiation', 'total_snowfall']]

storesVSstations = storesVSstations.rename(columns={"calendar_date": "visit_date"})


storesVSstations.to_pickle('../data/weather.pkl')

