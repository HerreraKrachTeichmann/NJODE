"""
author: Florian Krach & Calypso Herrera

code to train NJ-ODE on the physionet dataset provided by paper:
# Latent ODEs for Irregularly-Sampled Time Series
"""


# =====================================================================================================================
import torch
import argparse
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



# =====================================================================================================================
# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True
    N_CPUS = 3
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
    from ..latent_ODE import parse_datasets_LODE as parse_dataset
    from ..latent_ODE import likelihood_eval_LODE as likelihood_eval
except Exception:
    import NJODE.models as models
    import NJODE.data_utils as data_utils
    import latent_ODE.parse_datasets_LODE as parse_dataset
    import latent_ODE.likelihood_eval_LODE as likelihood_eval
import matplotlib.pyplot as plt


# =====================================================================================================================
# Global variables
data_path = data_utils.data_path
train_data_path = data_utils.training_data_path
saved_models_path = '{}saved_models_physionet/'.format(data_path)

METR_COLUMNS = ['epoch', 'train_time', 'eval_time', 'train_loss', 'eval_loss',
                'eval_metric', 'eval_metric_2']
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
        model_id=None, epochs=100, batch_size=50, save_every=1,
        learning_rate=0.001,
        hidden_size=41, bias=True, dropout_rate=0.1,
        ode_nn=default_ode_nn, readout_nn=default_readout_nn,
        enc_nn=default_enc_nn, use_rnn=False,
        solver="euler", weight=0.5, weight_decay=1.,
        dataset='physionet', saved_models_path=saved_models_path,
        quantization=0.016, n_samples=8000,
        eval_input_prob=None, eval_input_seed=3892,
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
    :param saved_models_path: str, where to save the models
    :param quantization: the time-step size in the physionet dataset (1=1h,
            0.016~1/60=1min)
    :param eval_input_prob: None or float in [0,1], the probability for each of
            the datapoints on the left out part of the eval set (second half of
            time points) to be used as input during eval. for the evaluation,
            the predicted value before this input is processed (i.e. before the
            jump) is used.
    :param eval_input_seed: None or int, seed for sampling from distribution,
            when deciding which data points are used from left-out part of eval
            set. If the seed is not None, in each call to the eval dataloader,
            the same points will be included additionally as input.
    :param options: kwargs, used keywords:
            'parallel'      bool, used by parallel_train.parallel_training
            'resume_training'   bool, used by parallel_train.parallel_training
            'which_loss'    'standard' or 'easy', used by models.NJODE
            'residual_enc_dec'  bool, whether resNNs are used for encoder and
                                readout NN, used by models.NJODE, default True
            'delta_t'       float, default equals quantization/48, which is the
                            step size when the time scale is normalized to
                            [0,1], to change stepsize of alg
            'load_best'     bool, whether to load the best checkpoint instead of
                            the last checkpoint when loading the model. Mainly
                            used for evaluating model at the best checkpoint.
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
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args_dict = vars(args)
    args_dict["dataset"] = "physionet"
    args_dict["n"] = n_samples
    args_dict["quantization"] = quantization
    args_dict["batch_size"] = batch_size
    args_dict["classif"] = False
    args_dict["eval_input_prob"] = eval_input_prob
    args_dict["eval_input_seed"] = eval_input_seed

    data_objects = parse_dataset.parse_datasets(args, device)

    dl = data_objects["train_dataloader"]
    dl_test = data_objects["test_dataloader"]
    input_size = data_objects["input_dim"]
    output_size = input_size
    T = 1 + 1e-12
    delta_t = quantization/48.
    if "delta_t" in options:
        delta_t = options['delta_t']

    # get params_dict
    params_dict = {
        'input_size': input_size, 'epochs': epochs,
        'hidden_size': hidden_size, 'output_size': output_size,
        'bias': bias,
        'ode_nn': ode_nn, 'readout_nn': readout_nn, 'enc_nn': enc_nn,
        'use_rnn': use_rnn,
        'dropout_rate': dropout_rate, 'batch_size': batch_size,
        'solver': solver, 'dataset': dataset,
        'quantization': quantization, 'n_samples': n_samples,
        'eval_input_prob': eval_input_prob,
        'learning_rate': learning_rate, 'eval_input_seed': eval_input_seed,
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
        model_name = 'NJ-ODE'
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

        # send notification
        if SEND:
            SBM.send_notification(
                text='start training on physionet: {} id={}'.format(
                    model_name, model_id)
            )
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
            b_size = b["batch_size"]
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
            else:
                raise ValueError("the other_model is not defined")
            loss.backward()
            optimizer.step()
        train_time = time.time() - t

        # -------- evaluation --------
        t = time.time()
        loss_val, mse_val, mse_val_2 = evaluate_model(
            model, dl_test, device, options, delta_t, T)
        eval_time = time.time() - t
        train_loss = loss.detach().numpy()

        print("epoch {}, weight={:.5f}, train-loss={:.5f}, "
              "eval-loss={:.5f}, eval-metric={:.5f}, "
              "eval-metric_2={:.5f}".format(
            model.epoch, model.weight, train_loss, loss_val,
            mse_val, mse_val_2))

        if mse_val < best_eval_metric:
            print('save new best model: last-best-metric: {:.5f}, '
                  'new-best-metric: {:.5f}, epoch: {}'.format(
                best_eval_metric, mse_val, model.epoch))
            models.save_checkpoint(model, optimizer, model_path_save_best,
                                   model.epoch)
            best_eval_metric = mse_val

        metric_app.append([model.epoch, train_time, eval_time, train_loss,
                           loss_val, mse_val, mse_val_2])

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

    # send notification
    if SEND and not skip_training:
        files_to_send = [model_metric_file]
        caption = "{} - id={}".format(model_name, model_id)
        SBM.send_notification(
            text='finished training on physionet: {}, id={}\n\n{}'.format(
                model_name, model_id, desc),
            files=files_to_send,
            text_for_files=caption
        )

    # delete model & free memory
    del model, dl, dl_test, data_objects
    gc.collect()

    return 0

def evaluate_model(model, dl_val, device, options, delta_t, T):
    with torch.no_grad():
        loss_val = 0
        num_obs = 0
        count = 0
        mse_val = 0
        mse_val_2 = 0
        model.eval()  # set model in evaluation mode
        for i, b in enumerate(dl_val):
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"].to(device)
            M = b["M"].to(device)
            obs_idx = b["obs_idx"]
            b_size = b["batch_size"]
            unique_idx, counts = np.unique(
                obs_idx.detach().numpy(), return_counts=True)
            n_obs_ot = np.zeros((b_size))
            n_obs_ot[unique_idx] = counts
            n_obs_ot = n_obs_ot.astype(np.int)
            n_obs_ot = torch.tensor(n_obs_ot).to(device)
            start_X = torch.tensor(
                np.zeros((b_size, X.size()[1])), dtype=torch.float32)

            if b["vals_val"] is not None:
                vals_val = b["vals_val"]
                mask_val = b["mask_val"]
                times_val = b["times_val"]

            if 'other_model' not in options:
                hT, e_loss, path_t, path_h, path_y = model(
                    times, time_ptr, X, obs_idx, delta_t, T, start_X,
                    n_obs_ot, until_T=True,
                    return_path=True, get_loss=True, M=M
                )
            else:
                raise ValueError

            time_indices = get_comparison_times_ind(path_t, times_val)
            path_y = path_y.detach().numpy()
            path_y = path_y[time_indices, :, :]
            path_y = np.transpose(path_y, (1, 0, 2))

            mse_loss = (((path_y - vals_val)**2) * mask_val).sum()
            loss_val += e_loss.detach().numpy()
            num_obs += mask_val.sum()
            count += 1
            mse_val += mse_loss
            mse_val_2 += torch.mean(
                likelihood_eval.compute_masked_likelihood(
                    torch.tensor(path_y, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(vals_val, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(mask_val, dtype=torch.float32).unsqueeze(0),
                    likelihood_eval.mse
                )
            ).detach().numpy()

        # TODO: it might be that latent ODE takes mean over all dimensions (not
        #  only observed ones) -> maybe better too use their functions to
        #  compute mse
        mse_val /= num_obs
        loss_val /= count
        mse_val_2 /= count
    return loss_val, mse_val, mse_val_2



def get_comparison_times_ind(path_t, times_val):
    """
    function to get the indices from path_t at which its entries are closest to
    the entries in times_val
    :param path_t: np.array of times (floats)
    :param times_val: np.array of times, s.t. each entry lies between
            np.min(path_t) and np.max(path_t)
    :return: array of indices of same length as times_val
    """

    assert np.min(path_t) < np.min(times_val) and \
           np.max(path_t)+1e-10 > np.max(times_val), \
        "mins: {}, {}, max: {}, {}".format(
            np.min(path_t), np.min(times_val),
            np.max(path_t), np.max(times_val)
        )

    indices = []
    for t in times_val:
        for i in range(len(path_t)-1):
            if abs(path_t[i]-t) < 1e-10 or path_t[i] <= t < path_t[i+1]:
                if abs(t - path_t[i]) <= abs(t - path_t[i+1]):
                    indices.append(i)
                else:
                    indices.append(i+1)
                break
            elif i == len(path_t)-2:
                indices.append(i + 1)
                break

    assert len(indices) == len(times_val)

    return indices



if __name__ == '__main__':

    pass