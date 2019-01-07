import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV



data = {
    'train': pd.read_pickle('../features/train_with_minmax_p2.pkl'),
    'test': pd.read_pickle('../features/test_with_minmax_p2.pkl')
}

x_train = data['train'].drop(['visitors','store_id','visit_date','skew_visitors_g','std_visitors_g',
                              'std_visitors_s_d','skew_visitors_s_d'], axis=1)
y_train = np.log1p(data['train']['visitors'].values)

x_test = data['test'].drop(['store_id','visit_date','skew_visitors_g','std_visitors_g',
                              'std_visitors_s_d','skew_visitors_s_d'], axis=1)

x_train['air_diff_date_mean'] = x_train['air_diff_date_mean'].astype('timedelta64[s]')
x_train['hpg_diff_date_mean'] = x_train['hpg_diff_date_mean'].astype('timedelta64[s]')
x_train['week_of_year'] = x_train['week_of_year'].astype(np.int64)

x_test['air_diff_date_mean'] = x_test['air_diff_date_mean'].astype('timedelta64[s]')
x_test['hpg_diff_date_mean'] = x_test['hpg_diff_date_mean'].astype('timedelta64[s]')
x_test['week_of_year'] = x_test['week_of_year'].astype(np.int64)

# Binding to float32
for c, dtype in zip(x_train.columns, x_train.dtypes):
    if dtype == np.float64:
        x_train[c] = x_train[c].astype(np.float32)

for c, dtype in zip(x_test.columns, x_test.dtypes):
    if dtype == np.float64:
        x_test[c] = x_test[c].astype(np.float32)


x_test = x_test[x_train.columns]

gbm = lgb.LGBMRegressor(
    boosting = 'gbdt',
    objective='regression',
    num_leaves=80,
    learning_rate=0.01,
    sub_feature = 0.8,
    min_data_in_leaf = 15,
    n_estimators=10000
    )


gbm.fit(x_train, y_train, eval_metric='rmse')
predict_y = gbm.predict(x_test)

y_test = pd.read_csv('../data/sample_submission.csv',engine='python')
y_test['visitors'] = np.expm1(predict_y)
y_test[['id','visitors']].to_csv(
    '0gbm_submission.csv', index=False, float_format='%.3f')


# Find the best parameters by GridSearchCV
#-----------------------------------------------------------------------------------------
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

# There are many parameters you can try.
# The more parameters you use, the more time you spend.
# parameters = {
#               'num_leaves' : [40, 60, 80],
#               'max_depth': [15, 20, 25],
#               'learning_rate': [0.008, 0.01, 0.02],
#               'feature_fraction': [0.6, 0.7, 0.8],
#               'sub_feature': [0.6, 0.7, 0.8],
#               'bagging_fraction': [0.6, 0.7, 0.8, 1],
#               'bagging_freq': [2, 4],
#               'min_data_in_leaf' : [5, 10, 15 , 20, 25, 30],
#               'lambda_l2': [0, 0.01, 0.1, 0.05, 0.5],
# }

# gbm = lgb.LGBMRegressor(
#     boosting = 'gbdt',
#     objective='regression',
#     n_estimators=10000
#     )

# gsearch = GridSearchCV(gbm, param_grid=parameters)
# gsearch.fit(x_train, y_train, eval_metric='rmse')
# print('Best parameters found by grid search are:', gsearch.best_params_)

# import math
# def rmsle(y, y_pred):
# 	assert len(y) == len(y_pred)
# 	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
# 	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

# predict_y = gsearch.predict(x_test)
# print('The rmsle of prediction is:', rmsle(y_test, np.expm1(predict_y)))
