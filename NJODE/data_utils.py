"""
author: Florian Krach & Calypso Herrera

data utilities for creating and loading synthetic test datasets
"""


# =====================================================================================================================
import numpy as np
import json, os, time, sys
from torch.utils.data import Dataset
import torch
import socket
import copy
import pandas as pd

sys.path.append("../")
try:
    import NJODE.stock_model as stock_model
except Exception:
    from . import stock_model


# =====================================================================================================================
hyperparam_default = {
    'drift': 2., 'volatility': 0.3, 'mean': 4,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 10000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1, 
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False, 'v0': 1,
}


_STOCK_MODELS = stock_model.STOCK_MODELS


data_path = '../data/'
training_data_path = '{}training_data/'.format(data_path)


# =====================================================================================================================
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_dataset_overview():
    data_overview = '{}dataset_overview.csv'.format(
        training_data_path)
    makedirs(training_data_path)
    if not os.path.exists(data_overview):
        df_overview = pd.DataFrame(
            data=None, columns=['name', 'id', 'description'])
    else:
        df_overview = pd.read_csv(data_overview, index_col=0)
    return df_overview, data_overview


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
    df_overview, data_overview = get_dataset_overview()

    np.random.seed(seed=seed)
    hyperparam_dict['model_name'] = stock_model_name
    obs_perc = hyperparam_dict['obs_perc']
    stockmodel = _STOCK_MODELS[stock_model_name](**hyperparam_dict)
    stock_paths, dt = stockmodel.generate_paths()
    size = stock_paths.shape
    observed_dates = np.random.random(size=(size[0], size[2]))
    observed_dates = (observed_dates < obs_perc)*1
    nb_obs = np.sum(observed_dates[:, 1:], axis=1)

    time_id = int(time.time())
    file_name = '{}-{}'.format(stock_model_name, time_id)
    path = '{}{}/'.format(training_data_path, file_name)
    desc = json.dumps(hyperparam_dict, sort_keys=True)
    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError
    df_app = pd.DataFrame(
        data=[[stock_model_name, time_id, desc]],
        columns=['name', 'id', 'description']
    )
    df_overview = pd.concat([df_overview, df_app],
                            ignore_index=True)
    df_overview.to_csv(data_overview)

    hyperparam_dict['dt'] = dt
    os.makedirs(path)
    with open('{}data.npy'.format(path), 'wb') as f:
        np.save(f, stock_paths)
        np.save(f, observed_dates)
        np.save(f, nb_obs)
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(hyperparam_dict, f, sort_keys=True)

    # stock_path dimension: [nb_paths, dimension, time_steps]
    return path, time_id


def create_combined_dataset(
        stock_model_names=("BlackScholes", "OrnsteinUhlenbeck"),
        hyperparam_dicts=(hyperparam_default, hyperparam_default),
        seed=0):
    """
    create a synthetic dataset using one of the stock-models
    :param stock_model_names: list of str, each str is a name of a stockmodel,
            see _STOCK_MODELS
    :param hyperparam_dicts: list of dict, each dict contains all needed
            parameters for the model
    :param seed: int, random seed for the generation of the dataset
    :return: str (path where the dataset is saved), int (time_id to identify
                the dataset)
    """
    df_overview, data_overview = get_dataset_overview()

    assert len(stock_model_names) == len(hyperparam_dicts)
    np.random.seed(seed=seed)

    # start to create paths from first model
    filename = 'combined_{}'.format(stock_model_names[0])
    maturity = hyperparam_dicts[0]['maturity']
    hyperparam_dicts[0]['model_name'] = stock_model_names[0]
    obs_perc = hyperparam_dicts[0]['obs_perc']
    stockmodel = _STOCK_MODELS[stock_model_names[0]](**hyperparam_dicts[0])
    stock_paths, dt = stockmodel.generate_paths()
    last = stock_paths[:, :, -1]

    # for every other model, add the paths created with this model starting at
    #   last point of previous model
    for i in range(1, len(stock_model_names)):
        dt_last = dt
        assert hyperparam_dicts[i]['dimension'] == \
               hyperparam_dicts[i-1]['dimension']
        assert hyperparam_dicts[i]['nb_paths'] == \
               hyperparam_dicts[i-1]['nb_paths']
        filename += '_{}'.format(stock_model_names[i])
        maturity += hyperparam_dicts[i]['maturity']
        hyperparam_dicts[i]['model_name'] = stock_model_names[i]
        obs_perc = hyperparam_dicts[0]['obs_perc']
        stockmodel = _STOCK_MODELS[stock_model_names[i]](**hyperparam_dicts[i])
        _stock_paths, dt = stockmodel.generate_paths(start_X=last)
        assert dt_last == dt
        last = _stock_paths[:, :, -1]
        stock_paths = np.concatenate(
            [stock_paths, _stock_paths[:, :, 1:]], axis=2
        )

    size = stock_paths.shape
    observed_dates = np.random.random(size=(size[0], size[2]))
    observed_dates = (observed_dates < obs_perc)*1
    nb_obs = np.sum(observed_dates[:, 1:], axis=1)

    time_id = int(time.time())
    file_name = '{}-{}'.format(filename, time_id)
    path = '{}{}/'.format(training_data_path, file_name)
    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError

    metadata = {'dt': dt, 'maturity': maturity,
                'dimension': hyperparam_dicts[0]['dimension'],
                'nb_paths': hyperparam_dicts[0]['nb_paths'],
                'model_name': 'combined',
                'stock_model_names': stock_model_names,
                'hyperparam_dicts': hyperparam_dicts}
    desc = json.dumps(metadata, sort_keys=True)

    df_app = pd.DataFrame(
        data=[[filename, time_id, desc]],
        columns=['name', 'id', 'description']
    )
    df_overview = pd.concat([df_overview, df_app],
                            ignore_index=True)
    df_overview.to_csv(data_overview)

    os.makedirs(path)
    with open('{}data.npy'.format(path), 'wb') as f:
        np.save(f, stock_paths)
        np.save(f, observed_dates)
        np.save(f, nb_obs)
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(metadata, f, sort_keys=True)

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
            if stock_model_name == ll.split('-')[0]:
                models.append(ll)
        times = [int(x.split('-')[1]) for x in models]
        if len(times) > 0:
            time_id = np.max(times)
        else:
            time_id = None
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

    res = {'times': np.array(times), 'time_ptr': np.array(time_ptr),
           'obs_idx': torch.tensor(obs_idx, dtype=torch.long),
           'start_X': start_X, 'n_obs_ot': nb_obs, 
           'X': torch.tensor(X, dtype=torch.float32),
           'true_paths': stock_paths, 'observed_dates': observed_dates}
    return res


def _get_func(name):
    """
    transform a function given as str to a python function
    :param name: str, correspond to a function,
            supported: 'exp', 'power-x' (x the wanted power)
    :return: numpy fuction
    """
    if name in ['exp', 'exponential']:
        return np.exp
    if 'power-' in name:
        x = float(name.split('-')[1])
        def pow(input):
            return np.power(input, x)
        return pow
    else:
        return None

def _get_X_with_func_appl(X, functions, axis):
    """
    apply a list of functions to the paths in X and append X by the outputs
    along the given axis
    :param X: np.array, with the data,
    :param functions: list of functions to be applied
    :param axis: int, the data_dimension (not batch and not time dim) along
            which the new paths are appended
    :return: np.array
    """
    Y = X
    for f in functions:
        Y = np.concatenate([Y, f(X)], axis=axis)
    return Y


def CustomCollateFnGen(func_names=None):
    """
    a function to get the costume collate function that can be used in
    torch.DataLoader with the wanted functions applied to the data as new
    dimensions
    -> the functions are applied on the fly to the dataset, and this additional
    data doesn't have to be saved
    :param func_names: list of str, with all function names, see _get_func
    :return: collate function, int (multiplication factor of dimension before
                and after applying the functions)
    """
    # get functions that should be applied to X, additionally to identity 
    functions = []
    if func_names is not None:
        for func_name in func_names:
            f = _get_func(func_name)
            if f is not None:
                functions.append(f)
    mult = len(functions) + 1

    def custom_collate_fn(batch):
        dt = batch[0]['dt']
        stock_paths = np.concatenate([b['stock_path'] for b in batch], axis=0)
        observed_dates = np.concatenate([b['observed_dates'] for b in batch],
                                        axis=0)
        nb_obs = torch.tensor(
            np.concatenate([b['nb_obs'] for b in batch], axis=0))

        # here axis=1, since we have elements of dim
        #    [batch_size, data_dimension] => add as new data_dimensions
        start_X = torch.tensor(
            _get_X_with_func_appl(stock_paths[:, :, 0], functions, axis=1), 
            dtype=torch.float32
        )
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
                        # here axis=0, since only 1 dim (the data_dimension),
                        #    i.e. the batch-dim is cummulated outside together
                        #    with the time dimension
                        X.append(_get_X_with_func_appl(stock_paths[i, :, t], 
                                                       functions, axis=0))
                        obs_idx.append(i)
                time_ptr.append(counter)

        assert len(obs_idx) == observed_dates[:, 1:].sum()

        res = {'times': np.array(times), 'time_ptr': np.array(time_ptr),
               'obs_idx': torch.tensor(obs_idx, dtype=torch.long),
               'start_X': start_X, 'n_obs_ot': nb_obs,
               'X': torch.tensor(X, dtype=torch.float32),
               'true_paths': stock_paths, 'observed_dates': observed_dates}
        return res

    return custom_collate_fn, mult











if __name__ == '__main__':
    # create_dataset()
    #
    # r = load_dataset()
    # for rr in r:
    #     print(rr)


    # dat = IrregularDataset(model_name='BlackScholes')
    # dl = DataLoader(dataset=dat, collate_fn=custom_collate_fn, shuffle=True, batch_size=5,
    #                 num_workers=1)
    # for b in dl:
    #     print(b)
    #     break


    # h = hyperparam_default
    # h['mean'] = 1
    # h['speed'] = 2
    # h['volatility'] = 3
    # h['return_vol'] = True
    # h['dimension'] = 2
    # h['nb_paths'] = 100
    # create_dataset('HestonWOFeller', hyperparam_dict=h)



    pass
