import numpy as np
import pandas as pd
from datetime import datetime

def get_day_of_week(dateString):
    return (datetime.strptime(dateString, "%Y-%m-%d").weekday())

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

data['tra'].to_pickle('visit_data.pkl')
