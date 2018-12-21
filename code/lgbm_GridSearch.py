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

x_train = data['train'].drop(['visitors','store_id','visit_date'], axis=1)
y_train = np.log1p(data['train']['visitors'].values)

x_test = data['test'].drop(['store_id','visit_date'], axis=1)


x_train['air_diff_date_mean'] = x_train['air_diff_date_mean'].astype('timedelta64[s]')
x_train['hpg_diff_date_mean'] = x_train['hpg_diff_date_mean'].astype('timedelta64[s]')
x_train['week_of_year'] = x_train['week_of_year'].astype(np.int64)

x_test['air_diff_date_mean'] = x_test['air_diff_date_mean'].astype('timedelta64[s]')
x_test['hpg_diff_date_mean'] = x_test['hpg_diff_date_mean'].astype('timedelta64[s]')
x_test['week_of_year'] = x_test['week_of_year'].astype(np.int64)

print('Binding to float32')
for c, dtype in zip(x_train.columns, x_train.dtypes):
    if dtype == np.float64:
        x_train[c] = x_train[c].astype(np.float32)

for c, dtype in zip(x_test.columns, x_test.dtypes):
    if dtype == np.float64:
        x_test[c] = x_test[c].astype(np.float32)


# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

x_test = x_test[x_train.columns]

# parameters = {
#               'num_leaves' : [40, 60, 80],
#               'max_depth': [10, 15, 20, 25, 35],
#               'learning_rate': [0.005, 0.008, 0.01,0.02],
#               'feature_fraction': [0.6, 0.8],
#               'bagging_fraction': [0.6, 0.8],
#               'bagging_freq': [2, 4]
#               'lambda_l2': [0, 0.0001, 0.01, 0.1, 0.05, 0.5, 10],
#               'metric' : ['rmse']
# }

gbm = lgb.LGBMRegressor(
    # boosting = 'gbdt',    # dart沒有比較好
    objective='regression',
    num_leaves=60,
    learning_rate=0.01,
    bagging_fraction = 0.6,
    feature_fraction = 0.8,
    sub_feature = 0.7,
    lambda_l2 = 0.01,
    max_depth = 20,
    # bagging_freq = 4,

    # num_threads = 4,
    # max_bin = 400,
    # min_hessian = 1,
    # verbose = -1
    n_estimators=10000
    )

# 調參數
# gsearch = GridSearchCV(gbm, param_grid=parameters)

# print("Best parameters set:")
# best_parameters = gsearch.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))

# predict_y = gsearch.predict(x_test)
# print('The rmsle of prediction is:', mean_squared_log_error(y_test, np.expm1(predict_y)))

# print("----------------")
# gbm.fit(x_train, y_train, eval_metric='rmse')
# predict_y = gbm.predict(x_test)
# print('The rmsle of prediction is:', mean_squared_log_error(y_test, np.expm1(predict_y)))

print("----------------")
gbm.fit(x_train, y_train, eval_metric='rmse')
predict_y = gbm.predict(x_test)

y_test = pd.read_csv('../data/sample_submission.csv')
y_test['visitors'] = np.expm1(predict_y)
y_test[['id', 'visitors']].to_csv('gbm0_submission.csv', index=False, float_format='%.3f')