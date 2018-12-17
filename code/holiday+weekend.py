import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

lbl = preprocessing.LabelEncoder()


data = {
    'hol':
    pd.read_csv('../data/date_info.csv')
    
}

data['hol']['calendar_date'] = pd.to_datetime(data['hol']['calendar_date'])
data['hol']['weekend_flg'] = data['hol']['day_of_week'].isin(['Saturday', 'Sunday'])*1
data['hol']['weekend&holiday_flg'] = data['hol']['weekend_flg'] | data['hol']['holiday_flg']
#data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol'] = data['hol'].drop(['day_of_week'], axis=1)
data['hol']['calendar_date'] = data['hol']['calendar_date'].dt.date

data['hol'] = data['hol'].rename(columns={"calendar_date": "visit_date"})


data['hol'].to_pickle('../data/weekend.pkl')

