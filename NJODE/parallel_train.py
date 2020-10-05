"""
author: Florian Krach & Calypso Herrera

coode for parallel training
"""

# =====================================================================================================================
import numpy as np
import os, sys
import pandas as pd
import json
import socket
import matplotlib

from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

# from telegram_notifications import send_bot_message as SBM

# =====================================================================================================================
# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    N_JOBS = 1
    SEND = False
else:
    SERVER = True
    N_JOBS = 12
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
except Exception:
    import NJODE.data_utils as data_utils
    import NJODE.train as train
    import NJODE.climate_train as climate_train


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
    elif params['dataset'] in ["BlackScholes", "Heston", "OrnsteinUhlenbeck"]:
        return train.train(**params)
    elif params['dataset'] in ['climate', 'Climate']:
        return climate_train.train(**params)
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


def parallel_training(params=None, model_ids=None, nb_jobs=1, first_id=None):
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
    :return:
    """
    model_overview_file_name = '{}model_overview.csv'.format(
        train.saved_models_path)
    train.makedirs(train.saved_models_path)
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
                params[i] = params_dict
            else:
                desc = json.dumps(param, sort_keys=True)
                df_ov_app = pd.DataFrame([[model_id, desc]],
                                         columns=['id', 'description'])
                df_overview = pd.concat([df_overview, df_ov_app], ignore_index=True)
                df_overview.to_csv(model_overview_file_name)
                param['model_id'] = model_id
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
                params.append(params_dict)

    for param in params:
        param['parallel'] = True

    # if SEND:
    #     SBM.send_notification(
    #         text='start parallel training controlled ODE-RNN\nparams:'
    #              '\n\n{}'.format(params)
    #     )

    try:
        results = Parallel(n_jobs=nb_jobs)(delayed(train_switcher)(**param)
                                           for param in params)
        # if SEND:
        #     SBM.send_notification(
        #         text='finished parallel training controlled ODE-RNN\nparams:'
        #              '\n\n{}'.format(params)
        #     )
    except Exception as e:
        if SEND:
            # SBM.send_notification(
            #     text='error in parallel training controlled ODE-RNN\nerror:'
            #          '\n\n{}'.format(e)
            # )
            pass
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
    training_size = [int(100*2**x) for x in np.linspace(1, 7, 7)]
    network_size = [int(5*2**x) for x in np.linspace(1, 6, 6)]
    ode_nn = [((size, 'tanh'), (size, 'tanh')) for size in network_size]
    params_list = []
    for _ode_nn in ode_nn:
        param_dict3 = {
            'epochs': [100],
            'batch_size': [20],
            'save_every': [5],
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
            'dataset': ["BlackScholes"],
            'dataset_id': [None],
            'plot': [True],
            'paths_to_plot': [(0, 1, 2, 3, 4,)],
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
    #                   first_id=1)


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
    # parallel training on climate dataset for cross validation
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









