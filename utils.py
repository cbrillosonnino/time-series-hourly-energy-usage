import yaml
import os 
import pickle
import requests

import pandas as pd
import numpy as np
import datetime as dt

import torch
from torch.utils.data import Dataset


def get_data(cred_path = 'credentials.yml'):

    cred = yaml.load(open(cred_path), Loader=yaml.FullLoader)
    API_key = cred['key']
    URL = 'http://api.eia.gov/series/?api_key={}&series_id=EBA.NY-ALL.D.HL'.format(API_key)
    r = requests.get(url = URL) 

    data = r.json() 
    data = pd.DataFrame(data['series'][0]['data'], columns = ['DateTime', 'MwH'])
    
    data['DateTime'] = data['DateTime'].apply(lambda x: x[:-3])
    data['DateTime'] =  pd.to_datetime(data['DateTime'])

    data.replace(0, np.nan, inplace = True)
    data['MwH'].interpolate(inplace = True)

    return data


def save_splits(directory = 'data', cred_path = 'credentials.yml'):

    data = get_data(cred_path)

    train = data[data['DateTime'] < dt.datetime(year = 2019, month = 1, day =1)].reset_index(drop=True)
    validation = data[(data['DateTime'] >= dt.datetime(year = 2019, month = 1, day =1)) &
                     (data['DateTime'] < dt.datetime(year = 2020, month = 1, day =1))].reset_index(drop=True)
    test = data[data['DateTime'] < dt.datetime(year = 2020, month = 1, day =1)].reset_index(drop=True)
    
    if not os.path.isdir(directory):
        os.mkdir(directory) 
        
    pickle.dump(train, open(os.path.join(directory,'train.p'), 'wb'))
    pickle.dump(validation, open(os.path.join(directory,'validation.p'), 'wb'))
    pickle.dump(test, open(os.path.join(directory,'test.p'), 'wb'))


class EnergyDemandDataset(Dataset):
    """Energy Demand Dataset Object"""

    def __init__(self, split, look_back, transform = 'min-max', directory = 'standard'):
        """
        Args:
            split (string): Data split -> train, validation, or test'.
            look_back (int): Number of time-steps into the past for each sample
            transform (string): Data normalization type -> min-max or standardize
            directory (string): Directory where data is located
        """
        
        assert split in ['train', 'validation', 'test'], "split needs to be either 'train', 'validation', or 'test'"
        assert transform in ['min-max', 'standard'], "transform only supports 'min-max' or 'standard' (standard score)"
        assert isinstance(look_back, int) and look_back > 0, 'look_back needs to be integer greater than 0'
        
        self.data = pickle.load(open(os.path.join(directory,'{}.p'.format(split)), 'rb'))['MwH'].values
        self.look_back = look_back
        self.transform = transform
        
        self.mean = 18251.291178385418
        self.std = 3475.1125327833797

        self.min = 11831.0
        self.max = 32076.0
        
        if self.transform == 'min-max':
            self.data = (self.data - self.min)/(self.max-self.min)
        elif self.transform == 'standard':
            self.data = (self.data - self.mean)/(self.std)
            
    def __len__(self):
        return self.data.shape[0]-self.look_back

    def __getitem__(self, idx):
        
        sequence = torch.Tensor(self.data[0+idx:self.look_back+idx])
        target = torch.Tensor([self.data[self.look_back+idx]])

        return sequence, target
