import numpy as np
import pandas as pd
import calendar
from datetime import datetime
from sklearn import preprocessing

def get_day_of_week(dateString):
    day_of_week = datetime.strptime(dateString, "%Y-%m-%d").weekday()
    return(calendar.day_name[day_of_week])

def get_month(dateString):
    return(datetime.strptime(dateString, "%Y-%m-%d").month)

def get_year(dateString):
    return(datetime.strptime(dateString, "%Y-%m-%d").year)

def get_week_of_year(dateString):
    return (datetime.strptime(dateString, "%Y-%m-%d").strftime("%W"))

def string_to_date(dateString):
    return datetime.strptime(dateString, "%Y-%m-%d")

data = {
    'tra': pd.read_csv('../data/air_visit_data.csv')
}

data['tra']['year'] = data['tra']['visit_date'].apply(get_year)
data['tra']['month'] = data['tra']['visit_date'].apply(get_month)
data['tra']['day_of_week'] = data['tra']['visit_date'].apply(get_day_of_week)
data['tra']['week_of_year'] = data['tra']['visit_date'].apply(get_week_of_year)
data['tra']['visit_date'] = data['tra']['visit_date'].apply(string_to_date)

lbl = preprocessing.LabelEncoder()
data['tra']['day_of_week'] = lbl.fit_transform(data['tra']['day_of_week'])

data['tra'].to_pickle('visit_data.pkl')

