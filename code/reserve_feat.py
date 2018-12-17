import pandas as pd
data_path = '../data/'
store_id_map = pd.read_csv(data_path + 'store_id_relation.csv').set_index('hpg_store_id',drop=False)
air_reserve = pd.read_csv(data_path + 'air_reserve.csv').rename(columns={'air_store_id':'store_id'})
hpg_reserve = pd.read_csv(data_path + 'hpg_reserve.csv').rename(columns={'hpg_store_id':'store_id'})

air_reserve['visit_date'] = pd.to_datetime(air_reserve['visit_datetime'].str[:10])
air_reserve['reserve_date'] = pd.to_datetime(air_reserve['reserve_datetime'].str[:10])

hpg_reserve['visit_date'] = pd.to_datetime(hpg_reserve['visit_datetime'].str[:10])
hpg_reserve['reserve_date'] = pd.to_datetime(hpg_reserve['reserve_datetime'].str[:10])
hpg_reserve['store_id'] = hpg_reserve['store_id'].map(store_id_map['air_store_id']).dropna()

def store_reserve_agg(reserve_df, a_h):
    names = {
        # sum and count of reserve visitors for each store on each day
        (a_h + '_reserve_visitors'): reserve_df['reserve_visitors'].sum(),
        (a_h + '_reserve_count'): reserve_df['reserve_visitors'].count(),
        # average of difference of time between reserve time & visit time for each store on each day
        (a_h + '_diff_date_mean'): reserve_df['diff_date'].mean(),
    }
    return pd.Series(names, index=[a_h+'_reserve_visitors', a_h+'_reserve_count', a_h+'_diff_date_mean'])

air_reserve['diff_date'] = air_reserve['visit_date'] - air_reserve['reserve_date']
air_store_reserve = air_reserve.groupby(['store_id', 'visit_date']).apply(store_reserve_agg, 'air')

hpg_reserve['diff_date'] = hpg_reserve['visit_date'] - hpg_reserve['reserve_date']
hpg_store_reserve = hpg_reserve.groupby(['store_id', 'visit_date']).apply(store_reserve_agg, 'hpg')

result = air_store_reserve.join(hpg_store_reserve, how='outer').fillna(-1).reset_index()
# print(result.head())
result.to_pickle('../features/reserve.pkl')
# features: store_id, visit_date, air_reserve_visitors, air_reserve_count, air_diff_date_mean, hpg_reserve_visitors, hpg_reserve_count, hpg_diff_date_mean