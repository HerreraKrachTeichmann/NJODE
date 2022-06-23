"""
author: Florian Krach & Calypso Herrera

code for parallel training
"""

# =====================================================================================================================
import numpy as np
import os, sys
import pandas as pd
import json
import socket
import matplotlib
import copy

from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    class Sbm:
        def __init__(self):
            pass
        @staticmethod
        def send_notification(text, *args, **kwargs):
            print(text)
    SBM = Sbm()

sys.path.append("../")
try:
    from . import extras as extras
except Exception:
    import NJODE.extras as extras

# =====================================================================================================================
# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    N_JOBS = 1
    SEND = False
else:
    SERVER = True
    N_JOBS = 27
    SEND = True
print(socket.gethostname())
print('SERVER={}'.format(SERVER))

if SERVER:
    matplotlib.use('Agg')

sys.path.append("../")
try:
    from . import data_utils as data_utils
    from . import train
    from . import climate_train
    from . import physionet_train
except Exception:
    import NJODE.data_utils as data_utils
    import NJODE.train as train
    import NJODE.climate_train as climate_train
    import NJODE.physionet_train as physionet_train


error_chat_id = "-437994211"
DEBUG = False

# =====================================================================================================================
# Functions
def train_switcher(**params):
    """
    function to call the correct train function depending on the dataset. s.t.
    parallel training easily works altough different fuctions need to be called
    :param params: all params needed by the train function, as passed by
            parallel_training
    :return: function call to the correct train function
    """
    if 'dataset' not in params:
        raise KeyError('the "dataset" needs to be specified')
    elif params['dataset'] in ["BlackScholes", "Heston", "OrnsteinUhlenbeck",
                               "HestonWOFeller", "sine_BlackScholes",
                               "sine_Heston", "sine_OrnsteinUhlenbeck"] or \
            'combined' in params['dataset']:
        return train.train(**params)
    elif params['dataset'] in ['climate', 'Climate']:
        return climate_train.train(**params)
    elif params['dataset'] in ['physionet', 'Physionet']:
        return physionet_train.train(**params)
    else:
        raise ValueError('the specified "dataset" is not supported')


def get_parameter_array(param_dict):
    """
    helper function to get a list of parameter-list with all combinations of
    parameters specified in a parameter-dict

    :param param_dict: dict with parameters
    :return: 2d-array with the parameter combinations
    """
    param_combs_dict_list = list(ParameterGrid(param_dict))
    return param_combs_dict_list


def parallel_training(params=None, model_ids=None, nb_jobs=1, first_id=None,
                      saved_models_path=train.saved_models_path,
                      overwrite_params=None):
    """
    function for parallel training, based on train.train
    :param params: a list of param_dicts, each dict corresponding to one model
            that should be trained, can be None if model_ids is given
            (then unused)
            all kwargs needed for train.train have to be in each dict
            -> giving the params together with first_id, they can be used to
                restart parallel training (however, the saved params for all
                models where the model_id already existed will be used instead
                of the params in this list, so that no unwanted errors are
                produced by mismatching. whenever a model_id didn't exist yet
                the params of the list are used to make a new one)
            -> giving params without first_id, all param_dicts will be used to
                initiate new models
    :param model_ids: list of ints, the model ids to use (only those for which a
            model was already initiated and its description was saved to the
            model_overview.csv file will be used)
            -> used to restart parallel training of certain model_ids after the
                training was stopped
    :param nb_jobs: int, the number of CPUs to use parallelly
    :param first_id: int or None, the model_id corresponding to the first
            element of params list
    :param saved_models_path: str, path to saved models
    :param overwrite_params: None or dict with key the param name to be
            overwritten and value the new value for this param. can bee  used to
            continue the training of a stored model, where some params should be
            changed (e.g. the number of epochs to train longer)
    :return:
    """
    if params is not None and 'saved_models_path' in params[0]:
        saved_models_path = params[0]['saved_models_path']
    model_overview_file_name = '{}model_overview.csv'.format(
        saved_models_path)
    train.makedirs(saved_models_path)
    if not os.path.exists(model_overview_file_name):
        df_overview = pd.DataFrame(data=None, columns=['id', 'description'])
        max_id = 0
    else:
        df_overview = pd.read_csv(model_overview_file_name, index_col=0)
        max_id = np.max(df_overview['id'].values)

    # get model_id, model params etc. for each param
    if model_ids is None and params is None:
        return 0
    if model_ids is None:
        if first_id is None:
            model_id = max_id + 1
        else:
            model_id = first_id
        for i, param in enumerate(params):
            if model_id in df_overview['id'].values:
                desc = (df_overview['description'].loc[
                    df_overview['id'] == model_id]).values[0]
                params_dict = json.loads(desc)
                params_dict['resume_training'] = True
                params_dict['model_id'] = model_id
                if overwrite_params:
                    for k, v in overwrite_params.items():
                        params_dict[k] = v
                    desc = json.dumps(params_dict, sort_keys=True)
                    df_overview.loc[
                        df_overview['id'] == model_id, 'description'] = desc
                    df_overview.to_csv(model_overview_file_name)
                params[i] = params_dict
            else:
                desc = json.dumps(param, sort_keys=True)
                df_ov_app = pd.DataFrame([[model_id, desc]],
                                         columns=['id', 'description'])
                df_overview = pd.concat([df_overview, df_ov_app],
                                        ignore_index=True)
                df_overview.to_csv(model_overview_file_name)
                params_dict = json.loads(desc)
                params_dict['resume_training'] = False
                params_dict['model_id'] = model_id
                params[i] = params_dict
            model_id += 1
    else:
        params = []
        for model_id in model_ids:
            if model_id not in df_overview['id'].values:
                print("model_id={} does not exist yet -> skip".format(model_id))
            else:
                desc = (df_overview['description'].loc[
                    df_overview['id'] == model_id]).values[0]
                params_dict = json.loads(desc)
                params_dict['model_id'] = model_id
                params_dict['resume_training'] = True
                if overwrite_params:
                    for k, v in overwrite_params.items():
                        params_dict[k] = v
                    desc = json.dumps(params_dict, sort_keys=True)
                    df_overview.loc[
                        df_overview['id'] == model_id, 'description'] = desc
                    df_overview.to_csv(model_overview_file_name)
                params.append(params_dict)

    for param in params:
        param['parallel'] = True

    if SEND:
        SBM.send_notification(
            text='start parallel training - \nparams:'
                 '\n\n{}'.format(params)
        )

    if DEBUG:
        results = Parallel(n_jobs=nb_jobs)(delayed(train_switcher)(**param)
                                           for param in params)
        if SEND:
            SBM.send_notification(
                text='finished parallel training - \nparams:'
                     '\n\n{}'.format(params)
            )
    else:
        try:
            results = Parallel(n_jobs=nb_jobs)(delayed(train_switcher)(**param)
                                               for param in params)
            if SEND:
                SBM.send_notification(
                    text='finished parallel training - \nparams:'
                         '\n\n{}'.format(params)
                )
        except Exception as e:
            if SEND:
                SBM.send_notification(
                    text='error in parallel training - \nerror:'
                         '\n\n{}'.format(e),
                    chat_id=error_chat_id
                )
            else:
                print('error:\n\n{}'.format(e))




if __name__ == '__main__':
    # ==========================================================================
    # create dataset
    # ==========================================================================
    #dataset_dict = data_utils.hyperparam_default
    #dataset_dict['nb_paths'] = 20000
    #for dataset in ["BlackScholes", "Heston", "OrnsteinUhlenbeck"]:
    #    datasetpath, dataset_id = data_utils.create_dataset(
    #        stock_model_name=dataset, hyperparam_dict=dataset_dict)


    # ==========================================================================
    # parallel training
    # ==========================================================================
    ode_nn = ((50, 'tanh'), (50, 'tanh'))
    readout_nn = ((50, 'tanh'), (50, 'tanh'))
    enc_nn = ((50, 'tanh'), (50, 'tanh'))
    param_dict1 = {
        'epochs': [200],
        'batch_size': [200],
        'save_every': [5],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [10],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [ode_nn],
        'readout_nn': [readout_nn],
        'enc_nn': [enc_nn],
        'use_rnn': [False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["BlackScholes", "Heston", "OrnsteinUhlenbeck"],
        'dataset_id': [None],
        'plot': [True],
        'paths_to_plot': [(0,1,2,3,4,)]
    }
    params_list1 = get_parameter_array(param_dict=param_dict1)

    # params_list = params_list1
    # print('combinations: {}'.format(len(params_list)))
    # nb_jobs = min(N_JOBS, len(params_list))
    # print('nb_jobs: {}'.format(nb_jobs))
    # parallel_training(params=params_list, model_ids=None, nb_jobs=nb_jobs,
    #                   first_id=4)

    # ==========================================================================
    # parallel training for convergence analysis
    # ==========================================================================
    # #1: define networks to train
    dataset = ["Heston"]
    path = '{}conv-study-Heston-saved_models/'.format(train.data_path)
    # dataset = ["BlackScholes"]
    # path = '{}conv-study-BS-saved_models/'.format(train.data_path)
    # dataset = ["OrnsteinUhlenbeck"]
    # path = '{}conv-study-OU-saved_models/'.format(train.data_path)

    training_size = [int(100 * 2 ** x) for x in np.linspace(1, 7, 7)]
    network_size = [int(5 * 2 ** x) for x in np.linspace(1, 6, 6)]
    ode_nn = [((size, 'tanh'), (size, 'tanh')) for size in network_size]
    params_list = []
    for _ode_nn in ode_nn:
        param_dict3 = {
            'epochs': [100],
            'batch_size': [20],
            'save_every': [10],
            'learning_rate': [0.001],
            'test_size': [0.2],
            'training_size': training_size,
            'seed': [398],
            'hidden_size': [10],
            'bias': [True],
            'dropout_rate': [0.1],
            'ode_nn': [_ode_nn],
            'readout_nn': [_ode_nn],
            'enc_nn': [_ode_nn],
            'use_rnn': [False],
            'func_appl_X': [[]],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'dataset': dataset,
            'dataset_id': [None],
            'plot': [True],
            'paths_to_plot': [(0,)],
            'saved_models_path': [path],
            'evaluate': [True]
        }
        params_list3 = get_parameter_array(param_dict=param_dict3)
        params_list += params_list3

    # # #2: parallel training
    # params_list = params_list * 5  # to get variance across different trials
    # print('combinations: {}'.format(len(params_list)))
    # nb_jobs = min(N_JOBS, len(params_list))
    # print('nb_jobs: {}'.format(nb_jobs))
    # parallel_training(params=params_list, model_ids=None, nb_jobs=nb_jobs,
    #                   first_id=1, saved_models_path=path)
    #
    # # #3: plot convergence study
    # extras.plot_convergence_study(
    #     path=path,
    #     x_axis="training_size", x_log=True, y_log=True)
    # extras.plot_convergence_study(
    #     path=path,
    #     x_axis="network_size", x_log=True, y_log=True)


    # ==========================================================================
    # parallel training for GRU-ODE-Bayes
    # ==========================================================================
    # #1: define networks to train
    params_list = []
    param_dict3 = {
        'epochs': [100],
        'batch_size': [20],
        'save_every': [5],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [50, 100],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [None],
        'readout_nn': [None],
        'enc_nn': [None],
        'use_rnn': [False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["BlackScholes", "Heston", "OrnsteinUhlenbeck"],
        'dataset_id': [None],
        'plot': [True],
        'paths_to_plot': [(0, 1, 2, 3, 4,)],
        'evaluate': [True],
        'other_model': ['GRU_ODE_Bayes'],
        'GRU_ODE_Bayes-impute': [True, False],
        'GRU_ODE_Bayes-logvar': [True, False],
        'GRU_ODE_Bayes-mixing': [0.0001, 0.5],
    }
    params_list3 = get_parameter_array(param_dict=param_dict3)
    params_list += params_list3

    # for comparison: NJ-ODE
    ode_nn = ((50, 'tanh'), (50, 'tanh'))
    param_dict4 = {
        'epochs': [100],
        'batch_size': [20],
        'save_every': [5],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [10],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [ode_nn],
        'readout_nn': [ode_nn],
        'enc_nn': [ode_nn],
        'use_rnn': [False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["BlackScholes", "Heston", "OrnsteinUhlenbeck"],
        'dataset_id': [None],
        'plot': [True],
        'paths_to_plot': [(0, 1, 2, 3, 4,)],
        'evaluate': [True]
    }
    params_list4 = get_parameter_array(param_dict=param_dict4)
    params_list += params_list4

    # # #2: parallel training
    # print('combinations: {}'.format(len(params_list)))
    # nb_jobs = min(N_JOBS, len(params_list))
    # print('nb_jobs: {}'.format(nb_jobs))
    # parallel_training(params=params_list, model_ids=None, nb_jobs=nb_jobs,
    #                   first_id=1)



    # ==========================================================================
    # parallel training on climate dataset
    # ==========================================================================
    params_list = []
    size = 50
    _ode_nn = ((size, 'tanh'), (size, 'tanh'))
    param_dict = {
        'epochs': [200],
        'batch_size': [100],
        'save_every': [1],
        'learning_rate': [0.001],
        'hidden_size': [10],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_ode_nn],
        'readout_nn': [_ode_nn],
        'enc_nn': [_ode_nn],
        'use_rnn': [False],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["climate"],
        'data_index': [0, 1, 2, 3, 4],
        'delta_t': [0.1]
    }
    params_list1 = get_parameter_array(param_dict=param_dict)
    params_list += params_list1


    size = 400
    _ode_nn = ((size, 'tanh'), (size, 'tanh'))
    param_dict = {
        'epochs': [200],
        'batch_size': [100],
        'save_every': [1],
        'learning_rate': [0.001],
        'hidden_size': [50],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_ode_nn],
        'readout_nn': [_ode_nn],
        'enc_nn': [_ode_nn],
        'use_rnn': [False],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["climate"],
        'data_index': [0, 1, 2, 3, 4],
        'delta_t': [0.1]
    }
    params_list1 = get_parameter_array(param_dict=param_dict)
    params_list += params_list1

    # for comparison: GRU-ODE-Bayes with suggested hyper-params
    param_dict = {
        'epochs': [50],
        'batch_size': [100],
        'save_every': [1],
        'learning_rate': [0.001],
        'hidden_size': [50],
        'bias': [True],
        'dropout_rate': [0.2],
        'ode_nn': [None],
        'readout_nn': [None],
        'enc_nn': [None],
        'use_rnn': [False],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["climate"],
        'data_index': [1],
        'delta_t': [0.1],
        'other_model': ['GRU_ODE_Bayes'],
        'GRU_ODE_Bayes-impute': [False],
        'GRU_ODE_Bayes-logvar': [True],
        'GRU_ODE_Bayes-mixing': [1e-4],
        'GRU_ODE_Bayes-p_hidden': [25],
        'GRU_ODE_Bayes-prep_hidden': [10],
        'GRU_ODE_Bayes-cov_hidden': [50],
    }
    params_list2 = get_parameter_array(param_dict=param_dict)

    # # #2: parallel training
    # print('combinations: {}'.format(len(params_list)))
    # nb_jobs = min(N_JOBS, len(params_list))
    # print('nb_jobs: {}'.format(nb_jobs))
    # parallel_training(params=params_list, model_ids=None, nb_jobs=nb_jobs,
    #                   first_id=101)



    # ==========================================================================
    # parallel training for Heston without Feller
    # ==========================================================================
    # # 0: create datasets
    # dataset = 'HestonWOFeller'
    # if data_utils._get_time_id(dataset) is None:
    #     hyperparam_dict = {
    #         'drift': 2., 'volatility': 3., 'mean': 1.,
    #         'speed': 2., 'correlation': 0.5, 'nb_paths': 20000, 'nb_steps': 100,
    #         'S0': 1, 'maturity': 1., 'dimension': 1,
    #         'obs_perc': 0.1,
    #         'scheme': 'euler', 'return_vol': False, 'v0': 0.5,
    #     }
    #     data_utils.create_dataset(
    #         stock_model_name=dataset, hyperparam_dict=hyperparam_dict)
    #     hyperparam_dict['return_vol'] = True
    #     hyperparam_dict['dimension'] = 2
    #     data_utils.create_dataset(
    #         stock_model_name=dataset, hyperparam_dict=hyperparam_dict)

    # 1: load the datasets
    df_overview, filename = data_utils.get_dataset_overview()
    data_ids = []
    for index, row in df_overview.iterrows():
        if 'HestonWOFeller' in row['name']:
            data_ids.append(row['id'])

    # 2: define hyper params
    params_list = []
    ode_nn = ((50, 'tanh'), (50, 'tanh'))
    param_dict1 = {
        'epochs': [200],
        'batch_size': [100],
        'save_every': [5],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [10],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [ode_nn],
        'readout_nn': [ode_nn],
        'enc_nn': [ode_nn],
        'use_rnn': [False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': [dataset],
        'dataset_id': data_ids,
        'plot': [True],
        'paths_to_plot': [(0, 1, 2, 3, 4,)],
        'evaluate': [True]
    }
    params_list1 = get_parameter_array(param_dict=param_dict1)
    params_list += params_list1

    # # #3: parallel training
    # print('combinations: {}'.format(len(params_list)))
    # nb_jobs = min(N_JOBS, len(params_list))
    # print('nb_jobs: {}'.format(nb_jobs))
    # parallel_training(params=params_list, model_ids=None, nb_jobs=nb_jobs,
    #                   first_id=1)


    # ==========================================================================
    # parallel training for Combined stock models
    # ==========================================================================
    # 1: create datasets
    combined_dataset = ["OrnsteinUhlenbeck", "BlackScholes"]
    dat_name = 'combined'
    for d in combined_dataset:
        dat_name += '_{}'.format(d)

    # # create dataset if it does not exist yet
    # if data_utils._get_time_id(dat_name) is None:
    #     print('create new combined dataset')
    #     hyperparam_dict = data_utils.hyperparam_default
    #     hyperparam_dict['nb_paths'] = 20000
    #     hyperparam_dict['nb_steps'] = 50
    #     hyperparam_dict['maturity'] = 0.5
    #     hyperparam_dict['mean'] = 10
    #     hyperparam_dicts = [hyperparam_dict] * len(combined_dataset)
    #     data_utils.create_combined_dataset(
    #         stock_model_names=combined_dataset,
    #         hyperparam_dicts=hyperparam_dicts
    #     )

    # 2: define hyper params
    params_list = []
    _ode_nn = ((100, 'tanh'), (100, 'tanh'))
    param_dict1 = {
        'epochs': [200],
        'batch_size': [100],
        'save_every': [20],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [10],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_ode_nn],
        'readout_nn': [_ode_nn],
        'enc_nn': [_ode_nn],
        'use_rnn': [False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': [dat_name],
        'plot': [True],
        'paths_to_plot': [(0, 1, 2, 3, 4,)],
        'evaluate': [True],
    }
    params_list1 = get_parameter_array(param_dict=param_dict1)
    params_list += params_list1

    # #3: parallel training
    # print('combinations: {}'.format(len(params_list)))
    # nb_jobs = min(N_JOBS, len(params_list))
    # print('nb_jobs: {}'.format(nb_jobs))
    # parallel_training(params=params_list, model_ids=None, nb_jobs=nb_jobs,
    #                   first_id=501)



    # ==========================================================================
    # parallel training on physionet dataset
    # ==========================================================================
    # ---------- means and stds over multiple runs ---------
    path = '{}saved_models_physionet_comparison/'.format(train.data_path)
    network_size = [50, 200]
    ode_nn = [((size, 'tanh'), (size, 'tanh')) for size in network_size]
    params_list = []
    for _ode_nn in ode_nn:
        param_dict = {
            'epochs': [175],
            'batch_size': [50],
            'save_every': [1],
            'learning_rate': [0.001],
            'hidden_size': [41],
            'bias': [True],
            'dropout_rate': [0.1],
            'ode_nn': [_ode_nn],
            'readout_nn': [_ode_nn],
            'enc_nn': [_ode_nn],
            'use_rnn': [False],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'dataset': ["physionet"],
            'quantization': [0.016],
            'n_samples': [8000],
            'saved_models_path': [path],
        }
        params_list1 = get_parameter_array(param_dict=param_dict)
        params_list += params_list1

    # params_list = params_list * 5
    # #2: parallel training
    # print('combinations: {}'.format(len(params_list)))
    # nb_jobs = min(N_JOBS, len(params_list))
    # print('nb_jobs: {}'.format(nb_jobs))
    # parallel_training(params=params_list, model_ids=None, nb_jobs=nb_jobs,
    #                   first_id=1, saved_models_path=path)


    # ==========================================================================
    # parallel training for sine stock models
    # ==========================================================================
    # # 1: create datasets
    # if data_utils._get_time_id('sine_BlackScholes') is None:
    #     mn = "sine_BlackScholes"
    #     for sc in [2 * np.pi, 4 * np.pi]:
    #         print('create sine dataset: model={}, '
    #               'sine_coeff={}'.format(mn, sc))
    #         hd = copy.deepcopy(data_utils.hyperparam_default)
    #         hd['sine_coeff'] = sc
    #         hd['nb_paths'] = 20000
    #         data_utils.create_dataset(mn, hd)

    # 2: load the datasets
    df_overview, filename = data_utils.get_dataset_overview()
    data_names = []
    data_ids = []
    for index, row in df_overview.iterrows():
        if 'sine_' in row['name']:
            data_names.append(row['name'])
            data_ids.append(row['id'])

    # 3: define hyper params
    path = '{}saved_models_sine/'.format(train.data_path)
    params_list = []
    _ode_nn = ((400, 'tanh'), (400, 'tanh'))
    for dat_name, dat_id in zip(data_names, data_ids):
        for _ode_nn in ode_nn:
            param_dict1 = {
                'epochs': [100],
                'batch_size': [100],
                'save_every': [10],
                'learning_rate': [0.001],
                'test_size': [0.2],
                'seed': [398],
                'hidden_size': [10],
                'bias': [True],
                'dropout_rate': [0.1],
                'ode_nn': [_ode_nn],
                'readout_nn': [_ode_nn],
                'enc_nn': [_ode_nn],
                'use_rnn': [False],
                'func_appl_X': [[]],
                'solver': ["euler"],
                'weight': [0.5],
                'weight_decay': [1.],
                'dataset': [dat_name],
                'dataset_id': [dat_id],
                'plot': [True],
                'paths_to_plot': [(0, 1, 2, 3, 4,)],
                'evaluate': [True],
                'saved_models_path': [path],
            }
            params_list1 = get_parameter_array(param_dict=param_dict1)
            params_list += params_list1

    # # #3: parallel training
    # print('combinations: {}'.format(len(params_list)))
    # nb_jobs = min(N_JOBS, len(params_list))
    # print('nb_jobs: {}'.format(nb_jobs))
    # parallel_training(params=params_list, model_ids=None, nb_jobs=nb_jobs,
    #                   first_id=1, saved_models_path=path)






