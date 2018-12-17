import pandas as pd
from sklearn import preprocessing

data = {
    'rsv': pd.read_pickle('../features/reserve.pkl'),
    'sif': pd.read_pickle('../features/store_info.pkl'),
    'vst': pd.read_pickle('../features/visit_data_p2.pkl'),
    'wtr': pd.read_pickle('../features/weather_p2.pkl'),
    'hol': pd.read_pickle('../features/weekend&holiday_flg_p2.pkl')
}

###change data['wtr']['visit_date'] to datetime type
data['wtr']['visit_date'] = pd.to_datetime(data['wtr']['visit_date'])
data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
###some column rename
data['sif'] = data['sif'].rename(columns={'air_store_id':'store_id'})
data['vst'] = data['vst'].rename(columns={'air_store_id':'store_id'})
data['wtr'] = data['wtr'].rename(columns={'air_store_id':'store_id'})

# do merge
train = pd.DataFrame()
train = data['vst']
#print train.head()
train = pd.merge(train, data['sif'], how='left', on=['store_id'])
#print train.head()
train = pd.merge(train, data['wtr'], how='left', on=['store_id', 'visit_date'])
#print train.head()
train = pd.merge(train, data['hol'], how='left', on=['visit_date'])
#print train.head()
train = pd.merge(train, data['rsv'], how='left', on=['store_id', 'visit_date'])
#print train.head()
train = train.fillna(-1)
#print train.head()
train.to_pickle("../features/train.pkl")
