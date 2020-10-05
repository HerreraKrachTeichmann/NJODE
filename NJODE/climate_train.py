"""
author: Florian Krach & Calypso Herrera

code to train NJ-ODE on the climate data as provided in the official
implementation of GRU-ODE-Bayes: https://github.com/edebrouwer/gru_ode_bayes.

Parts of this code are copied from GRU-ODE-Bayes implementation.
"""

# =====================================================================================================================
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os, sys
import pandas as pd
import json
import time
import socket
import matplotlib
import matplotlib.colors
from torch.backends import cudnn
import gc

# from telegram_notifications import send_bot_message as SBM


# =====================================================================================================================
# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True
    N_CPUS = 1
print(socket.gethostname())
print('SERVER={}'.format(SERVER))
SEND = False
if SERVER:
    SEND = True
    matplotlib.use('Agg')

sys.path.append("../")
try:
    from . import models as models
    from . import data_utils as data_utils
    from ..GRU_ODE_Bayes import data_utils_gru_ode_bayes as data_utils_gru
    from ..GRU_ODE_Bayes import models_gru_ode_bayes as models_gru_ode_bayes
except Exception:
    import NJODE.models as models
    import NJODE.data_utils as data_utils
    import GRU_ODE_Bayes.data_utils_gru_ode_bayes as data_utils_gru
    import GRU_ODE_Bayes.models_gru_ode_bayes as models_gru_ode_bayes
import matplotlib.pyplot as plt


# =====================================================================================================================
# Global variables
data_path = data_utils.data_path
train_data_path = data_utils.training_data_path
saved_models_path = '{}saved_models/'.format(data_path)

METR_COLUMNS = ['epoch', 'train_time', 'eval_time', 'train_loss', 'eval_loss',
                'eval_metric', 'test_loss', 'test_metric']
default_ode_nn = ((50, 'tanh'), (50, 'tanh'))
default_readout_nn = ((50, 'tanh'), (50, 'tanh'))
default_enc_nn = ((50, 'tanh'), (50, 'tanh'))

ANOMALY_DETECTION = False
N_DATASET_WORKERS = 0



# =====================================================================================================================
# Functions
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def train(
        model_id=None, epochs=100, batch_size=100, save_every=1,
        learning_rate=0.001,
        hidden_size=10, bias=True, dropout_rate=0.1,
        ode_nn=default_ode_nn, readout_nn=default_readout_nn,
        enc_nn=default_enc_nn, use_rnn=False,
        solver="euler", weight=1., weight_decay=0.999,
        data_index=0, dataset='climate',
        **options
):
    """
    training function for controlled ODE-RNN model (models.NJODE),
    the model is automatically saved in the model-save-path with the given
    model id, also all evaluations of the model are saved there
    :param model_id: None or int, the id to save (or load if it already exists)
            the model, if None: next biggest unused id will be used
    :param epochs: int, number of epochs to train, each epoch is one cycle
            through all (random) batches of the training data
    :param batch_size: int
    :param save_every: int, defined number of epochs after each of which the
            model is saved and plotted if wanted. whenever the model has a new
            best eval-loss it is also saved, independent of this number (but not
            plotted)
    :param learning_rate: float
    :param hidden_size: see models.NJODE
    :param bias: see models.NJODE
    :param dropout_rate: float
    :param ode_nn: see models.NJODE
    :param readout_nn: see models.NJODE
    :param enc_nn: see models.NJODE
    :param use_rnn: see models.NJODE
    :param solver: see models.NJODE
    :param weight: see models.NJODE
    :param weight_decay: see models.NJODE
    :param data_index: int in {0,..,4}, which index set to use
    :param options: kwargs, used keywords:
            'parallel'      bool, used by parallel_train.parallel_training
            'resume_training'   bool, used by parallel_train.parallel_training
            'which_loss'    'standard' or 'easy', used by models.NJODE
            'residual_enc_dec'  bool, whether resNNs are used for encoder and
                                readout NN, used by models.NJODE, default True
            'delta_t'       float, default 0.1, to change stepsize of alg
            'load_best'     bool, whether to load the best checkpoint instead of
                            the last checkpoint when loading the model. Mainly
                            used for evaluating model at the best checkpoint.
            'other_model'   one of {'GRU_ODE_Bayes'}; the specifieed model is
                            trained instead of the controlled ODE-RNN model.
                            Other options/inputs might change or loose their
                            effect. The saved_models_path is changed to
                            "{...}<model-name>-saved_models/" instead of
                            "{...}saved_models/".
                -> 'GRU_ODE_Bayes' has the following extra options with the
                    names 'GRU_ODE_Bayes'+<option_name>, for the following list
                    of possible choices for <options_name>:
                    '-mixing'   float, default: 0.0001, weight of the 2nd loss
                                term of GRU-ODE-Bayes
                    '-solver'   one of {"euler", "midpoint", "dopri5"}, default:
                                "euler"
                    '-impute'   bool, default: False,
                                whether to impute the last parameter
                                estimation of the p_model for the next ode_step
                                as input. the p_model maps (like the
                                readout_map) the hidden state to the
                                parameter estimation of the normal distribution.
                    '-logvar'   bool, default: True, wether to use logarithmic
                                (co)variace -> hardcodinng positivity constraint
                    '-full_gru_ode'     bool, default: True,
                                        whether to use the full GRU cell
                                        or a smaller version, see GRU-ODE-Bayes
                    '-p_hidden'         int, default: hidden_size, size of the
                                        inner hidden layer of the p_model
                    '-prep_hidden'      int, default: hidden_size, in the
                                        observational cell (i.e. jumps) a prior
                                        matrix multiplication transforms the
                                        input to have the size
                                        prep_hidden * input_size
                    '-cov_hidden'       int, default: hidden_size, size of the
                                        inner hidden layer of the covariate_map.
                                        the covariate_map is used as a mapping
                                        to get the initial h (for controlled
                                        ODE-RNN this is done by the encoder)
    """

    initial_print = ""
    options['masked'] = True

    if ANOMALY_DETECTION:
        torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(0)
        np.random.seed(0)
        cudnn.deterministic = True

    # set number of CPUs
    if SERVER:
        torch.set_num_threads(N_CPUS)

    # get the device for torch
    if torch.cuda.is_available():
        gpu_num = 0
        device = torch.device("cuda:{}".format(gpu_num))
        torch.cuda.set_device(gpu_num)
        initial_print += '\nusing GPU'
    else:
        device = torch.device("cpu")
        initial_print += '\nusing CPU'

    # get data
    csv_file_path = os.path.join(
        train_data_path, 'climate/small_chunked_sporadic.csv')
    train_idx = np.load(os.path.join(
        train_data_path,
        'climate/small_chunk_fold_idx_{}/train_idx.npy'.format(data_index)
        ), allow_pickle=True)
    val_idx = np.load(os.path.join(
        train_data_path,
        'climate/small_chunk_fold_idx_{}/val_idx.npy'.format(data_index)
        ), allow_pickle=True)
    test_idx = np.load(os.path.join(
        train_data_path,
        'climate/small_chunk_fold_idx_{}/test_idx.npy'.format(data_index)
        ), allow_pickle=True)
    validation = True
    val_options = {"T_val": 150, "max_val_samples": 3}

    data_train = data_utils_gru.ODE_Dataset(csv_file=csv_file_path,
                                            label_file=None,
                                            cov_file=None, idx=train_idx)
    data_val = data_utils_gru.ODE_Dataset(csv_file=csv_file_path,
                                          label_file=None,
                                          cov_file=None, idx=val_idx,
                                          validation=validation,
                                          val_options=val_options)
    data_test = data_utils_gru.ODE_Dataset(csv_file=csv_file_path,
                                           label_file=None,
                                           cov_file=None, idx=test_idx,
                                           validation=validation,
                                           val_options=val_options)

    # get data loaders
    dl = DataLoader(
        dataset=data_train, collate_fn=data_utils_gru.custom_collate_fn,
        shuffle=True, batch_size=batch_size, num_workers=N_DATASET_WORKERS)
    dl_val = DataLoader(
        dataset=data_val, collate_fn=data_utils_gru.custom_collate_fn,
        shuffle=True, batch_size=len(val_idx), num_workers=N_DATASET_WORKERS)
    dl_test = DataLoader(
        dataset=data_test, collate_fn=data_utils_gru.custom_collate_fn,
        shuffle=True, batch_size=len(test_idx), num_workers=N_DATASET_WORKERS)

    input_size = data_train.variable_num
    output_size = input_size
    T = 200
    delta_t = 0.1
    if "delta_t" in options:
        delta_t = options['delta_t']

    # get params_dict
    params_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size, 'output_size': output_size,
        'bias': bias,
        'ode_nn': ode_nn, 'readout_nn': readout_nn, 'enc_nn': enc_nn,
        'use_rnn': use_rnn,
        'dropout_rate': dropout_rate, 'batch_size': batch_size,
        'solver': solver, 'data_index': data_index,
        'learning_rate': learning_rate,
        'weight': weight, 'weight_decay': weight_decay,
        'options': options}
    desc = json.dumps(params_dict, sort_keys=True)

    # get overview file
    resume_training = False
    if ('parallel' in options and options['parallel'] is False) or \
            ('parallel' not in options):
        model_overview_file_name = '{}model_overview.csv'.format(
            saved_models_path
        )
        makedirs(saved_models_path)
        if not os.path.exists(model_overview_file_name):
            df_overview = pd.DataFrame(data=None,
                                       columns=['id', 'description'])
            max_id = 0
        else:
            df_overview = pd.read_csv(model_overview_file_name, index_col=0)
            max_id = np.max(df_overview['id'].values)

        # get model_id, model params etc.
        if model_id is None:
            model_id = max_id + 1
        if model_id not in df_overview['id'].values:
            initial_print += '\nnew model_id={}'.format(model_id)
            df_ov_app = pd.DataFrame([[model_id, desc]],
                                     columns=['id', 'description'])
            df_overview = pd.concat([df_overview, df_ov_app],
                                    ignore_index=True)
            df_overview.to_csv(model_overview_file_name)
        else:
            initial_print += '\nmodel_id already exists -> resume training'
            resume_training = True
            desc = (df_overview['description'].loc[
                df_overview['id'] == model_id]).values[0]
            params_dict = json.loads(desc)
            options = params_dict['options']
    initial_print += '\nmodel params:\n{}'.format(desc)
    if 'resume_training' in options and options['resume_training'] is True:
        resume_training = True

    # get all needed paths
    model_path = '{}id-{}/'.format(saved_models_path, model_id)
    makedirs(model_path)
    model_path_save_last = '{}last_checkpoint/'.format(model_path)
    model_path_save_best = '{}best_checkpoint/'.format(model_path)
    makedirs(model_path_save_last)
    makedirs(model_path_save_best)
    model_metric_file = '{}metric_id-{}.csv'.format(model_path, model_id)

    # get the model & optimizer
    if 'other_model' not in options:
        model = models.NJODE(**params_dict)
        model_name = 'controlled ODE-RNN'
    elif options['other_model'] == "GRU_ODE_Bayes":
        model_name = 'GRU-ODE-Bayes'
        # get parameters for GRU-ODE-Bayes model
        hidden_size = params_dict['hidden_size']
        mixing = 0.0001
        if 'GRU_ODE_Bayes-mixing' in options:
            mixing = options['GRU_ODE_Bayes-mixing']
        solver = 'euler'
        if 'GRU_ODE_Bayes-solver' in options:
            solver = options['GRU_ODE_Bayes-solver']
        impute = False
        if 'GRU_ODE_Bayes-impute' in options:
            impute = options['GRU_ODE_Bayes-impute']
        logvar = True
        if 'GRU_ODE_Bayes-logvar' in options:
            logvar = options['GRU_ODE_Bayes-logvar']
        full_gru_ode = True
        if 'GRU_ODE_Bayes-full_gru_ode' in options:
            full_gru_ode = options['GRU_ODE_Bayes-full_gru_ode']
        p_hidden = hidden_size
        if 'GRU_ODE_Bayes-p_hidden' in options:
            p_hidden = options['GRU_ODE_Bayes-p_hidden']
        prep_hidden = hidden_size
        if 'GRU_ODE_Bayes-prep_hidden' in options:
            prep_hidden = options['GRU_ODE_Bayes-prep_hidden']
        cov_hidden = hidden_size
        if 'GRU_ODE_Bayes-cov_hidden' in options:
            cov_hidden = options['GRU_ODE_Bayes-cov_hidden']

        model = models_gru_ode_bayes.NNFOwithBayesianJumps(
            input_size=params_dict['input_size'],
            hidden_size=params_dict['hidden_size'],
            p_hidden=p_hidden, prep_hidden=prep_hidden,
            bias=params_dict['bias'],
            cov_size=params_dict['input_size'], cov_hidden=cov_hidden,
            logvar=logvar, mixing=mixing,
            dropout_rate=params_dict['dropout_rate'],
            full_gru_ode=full_gru_ode, solver=solver, impute=impute,
        )
    else:
        raise ValueError(
            "Invalid argument for (option) parameter 'other_model'."
            "Please check docstring for correct use.")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=0.0005)

    # load saved model if wanted/possible
    best_eval_metric = np.infty
    metr_columns = METR_COLUMNS
    if resume_training:
        initial_print += '\nload saved model ...'
        try:
            if 'load_best' in options and options['load_best']:
                models.get_ckpt_model(model_path_save_best, model, optimizer,
                                      device)
            else:
                models.get_ckpt_model(model_path_save_last, model, optimizer,
                                      device)
            df_metric = pd.read_csv(model_metric_file, index_col=0)
            best_eval_metric = np.min(df_metric['eval_metric'].values)
            model.epoch += 1
            model.weight_decay_step()
            initial_print += '\nepoch: {}, weight: {}'.format(
                model.epoch, model.weight)
        except Exception as e:
            initial_print += '\nloading model failed -> initiate new model'
            initial_print += '\nException:\n{}'.format(e)
            resume_training = False
    if not resume_training:
        initial_print += '\ninitiate new model ...'
        df_metric = pd.DataFrame(columns=metr_columns)

    # ---------------- TRAINING ----------------
    skip_training = True
    if model.epoch <= epochs:
        skip_training = False

        # # send notification
        # if SEND:
        #     SBM.send_notification(
        #         text='start training climate: {} id={}'.format(
        #             model_name, model_id)
        #     )
        initial_print += '\n\nmodel overview:'
        print(initial_print)
        print(model, '\n')

        # compute number of parameters
        nr_params = 0
        for name, param in model.named_parameters():
            skip = False
            for p_name in ['gru_debug', 'classification_model']:
                if p_name in name:
                    skip = True
            if not skip:
                nr_params += param.nelement()
        print('# parameters={}\n'.format(nr_params))
        print('start training ...')

    metric_app = []
    while model.epoch <= epochs:
        t = time.time()
        model.train()  # set model in train mode (e.g. BatchNorm)
        for i, b in tqdm.tqdm(enumerate(dl)):
            optimizer.zero_grad()
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"].to(device)
            M = b["M"].to(device)
            obs_idx = b["obs_idx"]
            b_size = len(b["pat_idx"])
            unique_idx, counts = np.unique(
                obs_idx.detach().numpy(), return_counts=True)
            n_obs_ot = np.zeros((b_size))
            n_obs_ot[unique_idx] = counts
            n_obs_ot = n_obs_ot.astype(np.int)
            n_obs_ot = torch.tensor(n_obs_ot).to(device)
            start_X = torch.tensor(
                np.zeros((b_size, X.size()[1])), dtype=torch.float32)

            if 'other_model' not in options:
                hT, loss = model(
                    times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
                    return_path=False, get_loss=True, M=M
                )
            elif options['other_model'] == "GRU_ODE_Bayes":
                hT, loss, _, _ = model(
                    times, time_ptr, X, M, obs_idx, delta_t, T, start_X,
                    return_path=False, smoother=False
                )
            else:
                raise ValueError
            loss.backward()
            optimizer.step()
        train_time = time.time() - t

        # -------- evaluation --------
        t = time.time()
        loss_val, mse_val = evaluate_model(
            model, dl_val, device, options, delta_t, T)
        eval_time = time.time() - t
        train_loss = loss.detach().numpy()

        print("epoch {}, weight={:.5f}, train-loss={:.5f}, "
              "eval-loss={:.5f}, eval-metric={:.5f}".format(
            model.epoch, model.weight, train_loss, loss_val, mse_val))

        if mse_val < best_eval_metric:
            print('save new best model: last-best-metric: {:.5f}, '
                  'new-best-metric: {:.5f}, epoch: {}'.format(
                best_eval_metric, mse_val, model.epoch))
            models.save_checkpoint(model, optimizer, model_path_save_best,
                                   model.epoch)
            best_eval_metric = mse_val
        loss_test, mse_test = evaluate_model(
            model, dl_test, device, options, delta_t, T)
        print("test-loss={:.5f}, test-metric={:.5f}".format(
            loss_test, mse_test))

        metric_app.append([model.epoch, train_time, eval_time, train_loss,
                           loss_val, mse_val, loss_test, mse_test])

        # save model
        if model.epoch % save_every == 0:
            print('save model ...')
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            metric_app = []
            print('saved!')

        model.epoch += 1
        model.weight_decay_step()

    # # send notification
    # if SEND and not skip_training:
    #     files_to_send = [model_metric_file]
    #     caption = "{} - id={}".format(model_name, model_id)
    #     SBM.send_notification(
    #         text='finished training on climate: {}, id={}\n\n{}'.format(
    #             model_name, model_id, desc),
    #         files=files_to_send,
    #         text_for_files=caption
    #     )

    # delete model & free memory
    del model, dl, dl_val, dl_test, data_train, data_val, data_test
    gc.collect()

    return 0


def evaluate_model(model, dl_val, device, options, delta_t, T):
    with torch.no_grad():
        loss_val = 0
        num_obs = 0
        mse_val = 0
        model.eval()  # set model in evaluation mode
        for i, b in enumerate(dl_val):
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"].to(device)
            M = b["M"].to(device)
            obs_idx = b["obs_idx"]
            b_size = len(b["pat_idx"])
            unique_idx, counts = np.unique(
                obs_idx.detach().numpy(), return_counts=True)
            n_obs_ot = np.zeros((b_size))
            n_obs_ot[unique_idx] = counts
            n_obs_ot = n_obs_ot.astype(np.int)
            n_obs_ot = torch.tensor(n_obs_ot).to(device)
            start_X = torch.tensor(
                np.zeros((b_size, X.size()[1])), dtype=torch.float32)

            if b["X_val"] is not None:
                X_val     = b["X_val"].to(device)
                M_val     = b["M_val"].to(device)
                times_val = b["times_val"]
                times_idx = b["index_val"]

            if 'other_model' not in options:
                hT, e_loss, path_t, path_h, path_y = model(
                    times, time_ptr, X, obs_idx, delta_t, T, start_X,
                    n_obs_ot, until_T=True,
                    return_path=True, get_loss=True, M=M
                )
            elif options['other_model'] == "GRU_ODE_Bayes":
                hT, e_loss, class_pred, path_t, path_y, h_vec, _, _ = model(
                    times, time_ptr, X, M, obs_idx, delta_t, T, start_X,
                    return_path=True, smoother=False
                )
            else:
                raise ValueError
            t_vec = np.around(
                path_t, str(delta_t)[::-1].find(
                    '.')).astype(np.float32)  # Round floating points error in the time vector.
            p_val = data_utils_gru.extract_from_path(
                t_vec, path_y, times_val, times_idx)
            if 'other_model' not in options:
                m = p_val
            elif options['other_model'] == "GRU_ODE_Bayes":
                m, v = torch.chunk(p_val, 2, dim=1)
            mse_loss = (torch.pow(X_val - m, 2) * M_val).sum()

            loss_val += e_loss.detach().numpy()
            num_obs += M_val.sum().detach().numpy()
            mse_val += mse_loss.detach().numpy()

        loss_val /= num_obs
        mse_val /= num_obs
    return loss_val, mse_val











if __name__ == '__main__':

    ode_nn = ((50, 'tanh'), (50, 'tanh'))
    readout_nn = ((50, 'tanh'), (50, 'tanh'))
    enc_nn = ((50, 'tanh'), (50, 'tanh'))

    train(model_id=None, epochs=100, batch_size=100, save_every=1,
          learning_rate=0.001, test_size=0.2, seed=398,
          hidden_size=10, bias=True, dropout_rate=0.1,
          ode_nn=ode_nn, enc_nn=enc_nn,
          readout_nn=readout_nn, use_rnn=False,
          which_loss='standard', residual_enc_dec=True,
          solver="euler", weight=0.5, weight_decay=1.,
          data_index=0, )



    # train GRU-ODE-Bayes
    train(model_id=None, epochs=100, batch_size=100, save_every=1,
          learning_rate=0.001, test_size=0.2, seed=398,
          hidden_size=50, bias=True, dropout_rate=0.1,
          ode_nn=None, enc_nn=None,
          readout_nn=None, use_rnn=False,
          residual_enc_dec=True,
          solver="euler", weight=0.5, weight_decay=1.,
          data_index=0,
          other_model='GRU_ODE_Bayes',
          **{'GRU_ODE_Bayes-impute': True, 'GRU_ODE_Bayes-logvar': True,
             'GRU_ODE_Bayes-mixing': 0.0001}
          )

