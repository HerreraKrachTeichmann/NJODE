"""
author: Florian Krach & Calypso Herrera

data utilities for creating and loading synthetic test dataseets
"""


# =====================================================================================================================
import numpy as np
import json, os, time
from torch.utils.data import Dataset
import torch
import socket

try:
    import controlled_ODE_RNN.stock_model as stock_model
except Exception:
    import stock_model


# =====================================================================================================================
hyperparam_default = {
    'drift': 2., 'volatility': 0.3, 'mean': 4,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 10000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1, 
    'obs_perc': 0.1
}


_STOCK_MODELS = stock_model.STOCK_MODELS


data_path = '../data/'
training_data_path = '{}training_data/'.format(data_path)


# =====================================================================================================================
def create_dataset(
        stock_model_name="BlackScholes", 
        hyperparam_dict=hyperparam_default,
        seed=0):
    """
    create a synthetic dataset using one of the stock-models
    :param stock_model_name: str, name of the stockmodel, see _STOCK_MODELS
    :param hyperparam_dict: dict, contains all needed parameters for the model
    :param seed: int, random seed for the generation of the dataset
    :return: str (path where the dataset is saved), int (time_id to identify
                the dataset)
    """
    np.random.seed(seed=seed)
    hyperparam_dict['model_name'] = stock_model_name
    obs_perc = hyperparam_dict['obs_perc']
    stockmodel = _STOCK_MODELS[stock_model_name](**hyperparam_dict)
    stock_paths, dt = stockmodel.generate_paths()
    size = stock_paths.shape
    observed_dates = np.random.random(size=(size[0], size[2]))
    observed_dates = (observed_dates < obs_perc)*1
    nb_obs = np.sum(observed_dates, axis=1)

    time_id = int(time.time())
    file_name = '{}-{}'.format(stock_model_name, time_id)
    path = '{}{}/'.format(training_data_path, file_name)
    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError

    hyperparam_dict['dt'] =  dt
    os.makedirs(path)
    with open('{}data.npy'.format(path), 'wb') as f:
        np.save(f, stock_paths)
        np.save(f, observed_dates)
        np.save(f, nb_obs)
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(hyperparam_dict, f, sort_keys=True)

    return path, time_id


def _get_time_id(stock_model_name="BlackScholes", time_id=None):
    """
    if time_id=None, get the time id of the newest dataset with the given name
    :param stock_model_name: str
    :param time_id: None or int
    :return: int, time_id
    """
    if time_id is None:
        l = os.listdir(training_data_path)
        models = []
        for ll in l:
            if stock_model_name in ll:
                models.append(ll)
        times = [int(x.split('-')[1]) for x in models]
        time_id = np.max(times)
    return time_id


def load_metadata(stock_model_name="BlackScholes", time_id=None):
    """
    load the metadata of a dataset specified by its name and id
    :return: dict (with hyperparams of the dataset)
    """
    time_id = _get_time_id(stock_model_name=stock_model_name, time_id=time_id)
    path = '{}{}-{}/'.format(training_data_path, stock_model_name, int(time_id))
    with open('{}metadata.txt'.format(path), 'r') as f:
        hyperparam_dict = json.load(f)
    return hyperparam_dict


def load_dataset(stock_model_name="BlackScholes", time_id=None):
    """
    load a saved dataset by its name and id
    :param stock_model_name: str, name
    :param time_id: int, id
    :return: np.arrays of stock_paths, observed_dates, number_observations
                dict of hyperparams of the dataset
    """
    time_id = _get_time_id(stock_model_name=stock_model_name, time_id=time_id)
    path = '{}{}-{}/'.format(training_data_path, stock_model_name, int(time_id))

    with open('{}data.npy'.format(path), 'rb') as f:
        stock_paths = np.load(f)
        observed_dates = np.load(f)
        nb_obs = np.load(f)
    with open('{}metadata.txt'.format(path), 'r') as f:
        hyperparam_dict = json.load(f)

    return stock_paths, observed_dates, nb_obs, hyperparam_dict


class IrregularDataset(Dataset):
    """
    class for iterating over a dataset
    """
    def __init__(self, model_name, time_id=None, idx=None):
        stock_paths, observed_dates, nb_obs, hyperparam_dict = load_dataset(
            stock_model_name=model_name, time_id=time_id)
        if idx is None:
            idx = np.arange(hyperparam_dict['nb_paths'])
        self.metadata = hyperparam_dict
        self.stock_paths = stock_paths[idx]
        self.observed_dates = observed_dates[idx]
        self.nb_obs = nb_obs[idx]

    def __len__(self):
        return len(self.nb_obs)

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        # stock_path dimension: [BATCH_SIZE, DIMENSION, TIME_STEPS]
        return {"idx": idx, "stock_path": self.stock_paths[idx], 
                "observed_dates": self.observed_dates[idx], 
                "nb_obs": self.nb_obs[idx], "dt": self.metadata['dt']}


def custom_collate_fn(batch):
    """
    function used in torch.DataLoader to construct the custom training batch out
    of the batch of data (non-standard transformations can be applied here)
    :param batch: the input batch (as returned by IrregularDataset)
    :return: a batch (as dict) as needed for the training in train.train
    """
    dt = batch[0]['dt']
    stock_paths = np.concatenate([b['stock_path'] for b in batch], axis=0)
    observed_dates = np.concatenate([b['observed_dates'] for b in batch],
                                    axis=0)
    nb_obs = torch.tensor(np.concatenate([b['nb_obs'] for b in batch], axis=0))
    
    start_X = torch.tensor(stock_paths[:,:,0], dtype=torch.float32)
    X = []
    times = []
    time_ptr = [0]
    obs_idx = []
    current_time = 0.
    counter = 0
    for t in range(1, observed_dates.shape[1]):
        current_time += dt
        if observed_dates[:, t].sum() > 0:
            times.append(current_time)
            for i in range(observed_dates.shape[0]):
                if observed_dates[i, t] == 1:
                    counter += 1
                    X.append(stock_paths[i, :, t])
                    obs_idx.append(i)
            time_ptr.append(counter)

    assert len(obs_idx) == observed_dates[:,1:].sum()

    res = {'times': times, 'time_ptr': time_ptr, 
           'obs_idx': torch.tensor(obs_idx, dtype=torch.long),
           'start_X': start_X, 'n_obs_ot': nb_obs, 
           'X': torch.tensor(X, dtype=torch.float32),
           'true_paths': stock_paths, 'observed_dates': observed_dates}
    return res







