import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

data = {
    'train': pd.read_pickle('../features/train.pkl'),
    'test': pd.read_pickle('../features/test.pkl')
}


print('Binding to float32')
for c, dtype in zip( data['train'].columns,  data['train'].dtypes):
    if dtype == np.float64:
        data['train'][c] = data['train'][c].astype(np.float32)
    elif c in ['week_of_year', 'air_diff_date_mean', 'hpg_diff_date_mean']:
        data['train'][c] = data['train'][c].astype(np.int64)

for c, dtype in zip(data['test'].columns, data['test'].dtypes):
    if dtype == np.float64:
        data['test'][c] = data['test'][c].astype(np.float32)
    elif c in ['week_of_year', 'air_diff_date_mean', 'hpg_diff_date_mean']:
        data['test'][c] = data['test'][c].astype(np.int64)

train_x = data['train'].drop(['store_id', 'visit_date', 'visitors'], axis=1)
train_y = np.log1p(data['train']['visitors'].values)


test_x = data['test'].drop(['store_id', 'visit_date'], axis=1)

test_x = test_x[train_x.columns]

def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X.iloc[randomList], Y[randomList]

def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val

import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

# train_x, train_y = shuffle(train_x, train_y)
# train_x, train_y, val_x, val_y = splitData(train_x, train_y, 0.1)

# parameter tuning of lightgbm
# start from default setting
gbm0 = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=60,
    learning_rate=0.01,
    n_estimators=10000)

gbm0.fit(train_x, train_y, eval_metric='rmse')

#predict_y = gbm0.predict(val_x)
#rmsle(val_y, predict_y)

predict_y = gbm0.predict(test_x)

test_y = pd.read_csv('../data/sample_submission.csv')
test_y['visitors'] = np.expm1(predict_y)
test_y[['id', 'visitors']].to_csv(
    'gbm0_submission.csv', index=False, float_format='%.3f')


