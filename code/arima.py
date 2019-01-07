import pandas as pd
import numpy as np
import statsmodels.tsa.arima_model as smt
from tqdm import tqdm
# from pandas.plotting import autocorrelation_plot
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.stattools import acf
# import matplotlib.pyplot as plt

def get_best_model(ts):
    best_aic = np.inf 
    best_order = None
    best_model = None
    for p in range(0,6):
        for d in range(0,2):
            for q in range(0,6):
                try:
                    tmp_model = smt.ARIMA(ts, order=(p,d,q)).fit(method='mle', trend='nc', disp=-1)
                    tmp_aic = tmp_model.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (p, d, q)
                        best_model = tmp_model
                except: continue
    
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    print(best_model.summary())
    return best_model

train = pd.read_pickle('../features/train.pkl')
train = train[['store_id', 'visit_date', 'visitors']]
train_gp = train.groupby('store_id')

n_steps = 39
date_series = [d.strftime('%Y-%m-%d') for d in pd.date_range('2017-04-23', periods=n_steps, freq='D')]
submit_df = pd.DataFrame(columns=['id','visitors'])

for g in tqdm(train_gp.groups.items()):
    store_id = g[0]
    store_hist = train.loc[g[1]].drop(['store_id'], axis=1)
    print(store_hist.head())
    best_model = get_best_model(store_hist['visitors'])
    pred = best_model.forecast(steps=n_steps)

    submit_id = [store_id+'_'+ds for ds in date_series]
    pred_df = pd.DataFrame(data=submit_id, columns=['id'])
    pred_df['visitors'] = pred[0]
    submit_df = pd.concat([submit_df, pred_df])

submit_df.to_csv('../submissions/arima.csv', index=False)

# Compute distribution of lags that have hightest acf value
# This can help you select the range of p and q
# acf_count = np.zeros(0)
# for g in tqdm(train_gp.groups.items()):
#     store_id = g[0]
#     store_hist = train.loc[g[1]].drop(['store_id'], axis=1)
#     acf_value = acf(store_hist['visitors'], nlags=20)
#     acf_value[0] = 0
#     a = np.where(acf_value == max(acf_value))
#     acf_count = np.append(acf_count, a[0])
# unique, counts = np.unique(acf_count, return_counts=True)
# print(dict(zip(unique, counts)))

# Plot autocorrelation and print adfuller result
# g = train_gp.groups.items()[0]
# store_id = g[0]
# print(store_id)
# store_hist = train.loc[g[1]].drop(['store_id'], axis=1)
# autocorrelation_plot(store_hist['visitors'])
# plt.show()
# print(adfuller(store_hist['visitors'], autolag='AIC')[1])

# Plot prediction
# plt.plot(store_hist['visit_date'], store_hist['visitors'])
# submit_df = submit_df.set_index('visit_date')
# plt.plot(submit_df['visit_date'], submit_df['visitors'])
# plt.show()

# Some restaurants do not need to predict: (312)
# air_2703dcb33192b181
# air_b2d8bc9c88b85f96
# air_d0a7bd3339c3d12a
# air_d63cfa6d6ab78446
# air_cb083b4789a8d3a2
# air_229d7e508d9f1b5e
# air_cf22e368c1a71d53
# air_0ead98dd07e7a82a