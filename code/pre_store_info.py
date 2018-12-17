import pandas as pd
from sklearn import preprocessing

data = {
    'si': pd.read_csv('../data/air_store_info.csv')
}

lbl = preprocessing.LabelEncoder()
data['si']['air_genre_name'] = lbl.fit_transform(data['si']['air_genre_name'])
data['si']['air_area_name'] = lbl.fit_transform(data['si']['air_area_name'])

data['si'].to_pickle("../features/store_info.pkl")