import numpy as np
import pandas as pd
import calendar
from datetime import datetime


def get_day_of_week(dateString):
    day_of_week = datetime.strptime(dateString, "%Y-%m-%d").weekday()
    return(calendar.day_name[day_of_week])

def get_month(dateString):
    return(datetime.strptime(dateString, "%Y-%m-%d").month)

def get_year(dateString):
    return(datetime.strptime(dateString, "%Y-%m-%d").year)

def get_week_of_year(dateString):
    return (datetime.strptime(dateString, "%Y-%m-%d").strftime("%W"))


data = {
    'tra': pd.read_csv('../data/air_visit_data.csv')
}

data['tra']['year'] = data['tra']['visit_date'].apply(get_year)
data['tra']['month'] = data['tra']['visit_date'].apply(get_month)
data['tra']['day_of_week'] = data['tra']['visit_date'].apply(get_day_of_week)
data['tra']['week_of_year'] = data['tra']['visit_date'].apply(get_week_of_year)

data['tra'].to_pickle('visit_data.pkl')
