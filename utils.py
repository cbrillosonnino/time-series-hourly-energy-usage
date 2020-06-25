import yaml
import os 
import pickle
import requests

import pandas as pd
import numpy as np
import datetime as dt

def get_data(cred_path = 'credentials.yml'):

    cred = yaml.load(open(cred_path), Loader=yaml.FullLoader)
    API_key = cred['key']
    URL = 'http://api.eia.gov/series/?api_key={}&series_id=EBA.NY-ALL.D.HL'.format(API_key)
    r = requests.get(url = URL) 

    data = r.json() 
    data = pd.DataFrame(data['series'][0]['data'], columns = ['DateTime', 'MwH'])
    
    data['DateTime'] = data['DateTime'].apply(lambda x: x[:-3])
    data['DateTime'] =  pd.to_datetime(data['DateTime'])

    data.replace(0., np.nan, inplace = True)
    data['MwH'].interpolate(inplace = True)

    return data


def save_splits(directory = 'data', cred_path = 'credentials.yml'):

    data = get_data(cred_path)

    train = data[data['DateTime'] < dt.datetime(year = 2019, month = 1, day =1)].reset_index(drop=True)
    validation = data[(data['DateTime'] >= dt.datetime(year = 2019, month = 1, day =1)) &
                     (data['DateTime'] < dt.datetime(year = 2020, month = 1, day =1))].reset_index(drop=True)
    test = data[data['DateTime'] < dt.datetime(year = 2020, month = 1, day =1)].reset_index(drop=True)
    
    if not os.path.isfile(directory):
        os.mkdir(directory) 
        
    pickle.dump(train, open(os.path.join(directory,'train.p'), 'wb'))
    pickle.dump(validation, open(os.path.join(directory,'validation.p'), 'wb'))
    pickle.dump(test, open(os.path.join(directory,'test.p'), 'wb'))
