import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

data = {
    'reserve': pd.read_pickle('../features/reserve.pkl'),
    'store_info': pd.read_pickle('../features/store_info.pkl'),
    'visit_data': pd.read_pickle('../features/visit_data.pkl'),
    'weather': pd.read_pickle('../features/weather.pkl'),
    'weekend': pd.read_pickle('../features/weekend.pkl'),
    'sample_submission': pd.read_csv('../data/sample_submission.csv'),
    'train': pd.read_pickle('../train.pkl')
}

data['sample_submission']['visit_date'] = data['sample_submission']['id'].map(lambda x: str(x).split('_')[2])
data['sample_submission']['air_store_id'] = data['sample_submission']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['sample_submission']['visit_date'] = pd.to_datetime(data['sample_submission']['visit_date'])
data['sample_submission'] = data['sample_submission'].drop(columns = ['id', 'visitors'])

data['sample_submission']['day_of_week'] = data['sample_submission']['visit_date'].dt.dayofweek
data['sample_submission']['year'] = data['sample_submission']['visit_date'].dt.year
data['sample_submission']['month'] = data['sample_submission']['visit_date'].dt.month
data['sample_submission']['week_of_year'] = data['sample_submission']['visit_date'].dt.weekofyear


data['weather']['visit_date'] = pd.to_datetime(data['weather']['visit_date'])
data['weekend']['visit_date'] = pd.to_datetime(data['weekend']['visit_date'])

data['reserve'] = data['reserve'].rename(columns={'store_id':'air_store_id'})

# do merge
test = data['sample_submission']
test = pd.merge(test, data['store_info'], how='left', on=['air_store_id'])
test = pd.merge(test, data['weekend'], how='left', on=['visit_date'])
test = pd.merge(test, data['weather'], how='left', on=['air_store_id', 'visit_date'])
test = pd.merge(test, data['reserve'], how='left', on=['air_store_id', 'visit_date'])
test = test.fillna(-1)

test = test.rename(columns={'air_store_id':'store_id'})

test.to_pickle('drive/Colab Notebooks/features/test.pkl')
test.to_pickle('drive/Colab Notebooks/features/test_p2.pkl', protocol = 2)

