import numpy as np
import pandas as pd
from sklearn import preprocessing

lbl = preprocessing.LabelEncoder()

# Data wrangling brought to you by the1owl
# https://www.kaggle.com/the1owl/surprise-me

data = {
    'hol':
    pd.read_csv('../data/date_info.csv')
}

data['hol']['calendar_date'] = pd.to_datetime(data['hol']['calendar_date'])
data['hol']['weekend_flg'] = data['hol']['day_of_week'].isin(['Saturday', 'Sunday'])*1
data['hol']['weekend&holiday_flg'] = data['hol']['weekend_flg'] | data['hol']['holiday_flg']
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['calendar_date'] = data['hol']['calendar_date'].dt.date

data['hol'].to_pickle("../features/weekend&holiday_flg.pkl")

