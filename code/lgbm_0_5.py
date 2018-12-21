import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb


data = {
    'train': pd.read_pickle('../features/train.pkl'),
    'test': pd.read_pickle('../features/test.pkl')
}

tmp = data['train'].groupby(
    ['store_id', 'day_of_week'],
    as_index=False)['visitors'].min().rename(columns={
        'visitors': 'min_visitors'
    })
data['train'] = pd.merge(data['train'], tmp, how='left', on=['store_id', 'day_of_week'])
data['test'] = pd.merge(data['test'], tmp, how='left', on=['store_id', 'day_of_week'])

tmp = data['train'].groupby(
    ['store_id', 'day_of_week'],
    as_index=False)['visitors'].mean().rename(columns={
        'visitors': 'mean_visitors'
    })
data['train'] = pd.merge(data['train'], tmp, how='left', on=['store_id', 'day_of_week'])
data['test'] = pd.merge(data['test'], tmp, how='left', on=['store_id', 'day_of_week'])

tmp = data['train'].groupby(
    ['store_id', 'day_of_week'],
    as_index=False)['visitors'].median().rename(columns={
        'visitors': 'median_visitors'
    })
data['train'] = pd.merge(data['train'], tmp, how='left', on=['store_id', 'day_of_week'])
data['test'] = pd.merge(data['test'], tmp, how='left', on=['store_id', 'day_of_week'])

tmp = data['train'].groupby(
    ['store_id', 'day_of_week'],
    as_index=False)['visitors'].max().rename(columns={
        'visitors': 'max_visitors'
    })
data['train'] = pd.merge(data['train'], tmp, how='left', on=['store_id', 'day_of_week'])
data['test'] = pd.merge(data['test'], tmp, how='left', on=['store_id', 'day_of_week'])

tmp = data['train'].groupby(
    ['store_id', 'day_of_week'],
    as_index=False)['visitors'].count().rename(columns={
        'visitors': 'count_observations'
    })
data['train'] = pd.merge(data['train'], tmp, how='left', on=['store_id', 'day_of_week'])
data['test'] = pd.merge(data['test'], tmp, how='left', on=['store_id', 'day_of_week'])

x_train = data['train'].drop(['visitors','store_id','visit_date'], axis=1)
y_train = np.log1p(data['train']['visitors'].values)

x_test = data['test'].drop(['store_id','visit_date'], axis=1)

x_train['air_diff_date_mean'] = x_train['air_diff_date_mean'].astype('timedelta64[s]')
x_train['hpg_diff_date_mean'] = x_train['hpg_diff_date_mean'].astype('timedelta64[s]')
x_train['week_of_year'] = x_train['week_of_year'].astype(np.int64)

x_test['air_diff_date_mean'] = x_test['air_diff_date_mean'].astype('timedelta64[s]')
x_test['hpg_diff_date_mean'] = x_test['hpg_diff_date_mean'].astype('timedelta64[s]')
x_test['week_of_year'] = x_test['week_of_year'].astype(np.int64)

for c, dtype in zip(x_train.columns, x_train.dtypes):
    if dtype == np.float64:
        x_train[c] = x_train[c].astype(np.float32)

for c, dtype in zip(x_test.columns, x_test.dtypes):
    if dtype == np.float64:
        x_test[c] = x_test[c].astype(np.float32)


x_test = x_test[x_train.columns]

gbm = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=60,
    learning_rate=0.01,
    n_estimators=10000
)

print("----------------")
gbm.fit(x_train, y_train, eval_metric='rmse')
predict_y = gbm.predict(x_test)

y_test = pd.read_csv('../data/sample_submission.csv')
y_test['visitors'] = np.expm1(predict_y)
y_test[['id', 'visitors']].to_csv('gbm0_submission.csv', index=False, float_format='%.3f')

