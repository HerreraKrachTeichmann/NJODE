"""
author: Florian Krach & Calypso Herrera

implementation of the trainig (and evaluation) of ODE-RNN
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

try:
    import models as models
    import data_utils as data_utils
except Exception:
    import controlled_ODE_RNN.models as models
    import controlled_ODE_RNN.data_utils as data_utils
import matplotlib.pyplot as plt


# =====================================================================================================================
# Global variables
N_CPUS = 1

data_path = data_utils.data_path
saved_models_path = '{}saved_models/'.format(data_path)

metr_columns = ['epoch', 'train_time', 'eval_time', 'train_loss', 'eval_loss',
                'optimal_eval_loss']
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
        learning_rate=0.001, test_size=0.2, seed=398,
        hidden_size=10, bias=True, dropout_rate=0.1,
        ode_nn=default_ode_nn, readout_nn=default_readout_nn,
        enc_nn=default_enc_nn, use_rnn=False,
        solver="euler", weight=0.5, weight_decay=1.,
        dataset='BlackScholes', dataset_id=None, plot=True, paths_to_plot=(0,),
        **options
):
    """
    training function for controlled ODE-RNN model (models.CODERNN),
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
    :param test_size: float in (0,1), the percentage of samples to use for the
            test set (here there exists only a test set, since there is no extra
            evaluation)
    :param seed: int, seed for the random splitting of the dataset into train
            and test
    :param hidden_size: see models.CODERNN
    :param bias: see models.CODERNN
    :param dropout_rate: float
    :param ode_nn: see models.CODERNN
    :param readout_nn: see models.CODERNN
    :param enc_nn: see models.CODERNN
    :param use_rnn: see models.CODERNN
    :param solver: see models.CODERNN
    :param weight: see models.CODERNN
    :param weight_decay: see models.CODERNN
    :param dataset: str, which dataset to use, supported: {'BlackScholes',
            'Heston', 'OrnsteinUhlenbeck'}. The corresponding dataset already
            needs to exist (create it first using data_utils.create_dataset)
    :param dataset_id: int or None, the id of the dataset to be use, if None,
            the latest generated dataset of the given name will be used
    :param plot: bool, whethere to plot
    :param paths_to_plot: list of ints, which paths of the test-set should be
            plotted
    :param options: kwargs, used keywords:
            'plot_only'     bool, whether the model is used only to plot after
                            initiating or loading (i.e. no training) and exit
                            afterwards (used by demo)
            'which_loss'    'standard' or 'easy', used by models.CODERNN
            'residual_enc_dec'  bool, whether resNNs are used for encoder and
                                readout NN, used by models.CODERNN
    """

    if ANOMALY_DETECTION:
        torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(0)
        np.random.seed(0)
        cudnn.deterministic = True

    # set number of CPUs
    torch.set_num_threads(N_CPUS)

    # get the device for torch
    if torch.cuda.is_available():
        gpu_num = 0
        device = torch.device("cuda:{}".format(gpu_num))
        torch.cuda.set_device(gpu_num)
        print('using GPU')
    else:
        device = torch.device("cpu")
        print('using CPU')

    # load dataset and dataset-metadata
    dataset_id = int(data_utils._get_time_id(stock_model_name=dataset,
                                             time_id=dataset_id))
    dataset_metadata = data_utils.load_metadata(stock_model_name=dataset, 
                                                time_id=dataset_id)
    input_size = dataset_metadata['dimension']
    output_size = input_size
    T = dataset_metadata['maturity']
    delta_t = dataset_metadata['dt']
    
    # load raw data
    train_idx, val_idx = train_test_split(
        np.arange(dataset_metadata["nb_paths"]), test_size=test_size,
        random_state=seed
    )
    data_train = data_utils.IrregularDataset(
        model_name=dataset, time_id=dataset_id, idx=train_idx)
    data_val = data_utils.IrregularDataset(
        model_name=dataset, time_id=dataset_id, idx=val_idx)
    
    collate_fn = data_utils.custom_collate_fn
    dl = DataLoader(
        dataset=data_train, collate_fn=collate_fn,
        shuffle=True, batch_size=batch_size, num_workers=N_DATASET_WORKERS)
    dl_val = DataLoader(
        dataset=data_val, collate_fn=collate_fn,
        shuffle=False, batch_size=len(data_val), num_workers=N_DATASET_WORKERS)
    
    # get optimal eval loss
    stockmodel = data_utils._STOCK_MODELS[
        dataset_metadata['model_name']](**dataset_metadata)
    opt_eval_loss = compute_optimal_eval_loss(
        dl_val, stockmodel, delta_t, T)
    print('optimal eval loss (achieved by true cond exp): {:.5f}'.format(
        opt_eval_loss))

    # get params_dict
    params_dict = {
        'input_size': input_size,
        'hidden_size': hidden_size, 'output_size': output_size, 'bias': bias,
        'ode_nn': ode_nn, 'readout_nn': readout_nn, 'enc_nn': enc_nn, 
        'use_rnn': use_rnn,
        'dropout_rate': dropout_rate, 'batch_size': batch_size,
        'solver': solver, 'dataset': dataset, 'dataset_id': dataset_id,
        'learning_rate': learning_rate, 'test_size': test_size, 'seed': seed,
        'weight': weight, 'weight_decay': weight_decay,
        'optimal_eval_loss': opt_eval_loss, 'options': options}
    desc = json.dumps(params_dict, sort_keys=True)

    # get overview file
    resume_training = False
    model_overview_file_name = '{}model_overview.csv'.format(saved_models_path)
    makedirs(saved_models_path)
    if not os.path.exists(model_overview_file_name):
        df_overview = pd.DataFrame(data=None, columns=['id', 'description'])
        max_id = 0
    else:
        df_overview = pd.read_csv(model_overview_file_name, index_col=0)
        max_id = np.max(df_overview['id'].values)

    # get model_id, model params etc.
    if model_id is None:
        model_id = max_id + 1
    if model_id not in df_overview['id'].values:
        print('new model_id={}'.format(model_id))
        df_ov_app = pd.DataFrame([[model_id, desc]],
                                 columns=['id', 'description'])
        df_overview = pd.concat([df_overview, df_ov_app], ignore_index=True)
        df_overview.to_csv(model_overview_file_name)
    else:
        print('model_id already exists -> resume training')
        resume_training = True
        desc = (df_overview['description'].loc[df_overview['id'] == model_id]).values[0]
        params_dict = json.loads(desc)
    print('model params:\n{}'.format(desc))

    # get all needed paths
    model_path = '{}id-{}/'.format(saved_models_path, model_id)
    makedirs(model_path)
    model_path_save_last = '{}last_checkpoint/'.format(model_path)
    model_path_save_best = '{}best_checkpoint/'.format(model_path)
    makedirs(model_path_save_last)
    makedirs(model_path_save_best)
    model_metric_file = '{}metric_id-{}.csv'.format(model_path, model_id)
    
    # get the model & optimizer
    model = models.CODERNN(**params_dict)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=0.0005)
    
    # load saved model if wanted/possible
    best_eval_loss = np.infty
    if resume_training:
        print('load saved model ...')
        try:
            models.get_ckpt_model(model_path_save_last, model, optimizer,
                                  device)
            df_metric = pd.read_csv(model_metric_file, index_col=0)
            best_eval_loss = np.min(df_metric['eval_loss'].values)
            model.epoch += 1
            model.weight_decay_step()
            print('epoch: {}, weight: {}'.format(model.epoch, model.weight))
        except Exception as e:
            print('loading model failed -> initiate new model')
            print('Exception:\n{}'.format(e))
            resume_training = False
    if not resume_training:
        print('initiate new model ...')
        df_metric = pd.DataFrame(columns=metr_columns)

    # ---------- plot only option for demo ------------
    if 'plot_only' in options and options['plot_only']:
        for i, b in enumerate(dl_val):
                batch = b
        print('plotting ...')
        plot_save_path = '{}plots/'.format(model_path)
        plot_filename = 'epoch-{}'.format(model.epoch)
        plot_filename = plot_filename + '_path-{}.pdf'
        curr_opt_loss = plot_one_path_with_pred(
            device, model, batch, stockmodel, delta_t, T,
            path_to_plot=paths_to_plot, save_path=plot_save_path,
            filename=plot_filename
        )
        print('optimal eval-loss (with current weight={:.5f}): '
              '{:.5f}'.format(model.weight, curr_opt_loss))
        return 0

    # ---------------- TRAINING ----------------
    print('start trainig ...')
    metric_app = []
    while model.epoch <= epochs:
        t = time.time()
        model.train()  # set model in train mode (e.g. BatchNorm)
        for i, b in tqdm.tqdm(enumerate(dl)):
            optimizer.zero_grad()
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"].to(device)
            start_X = b["start_X"].to(device)
            obs_idx = b["obs_idx"]
            n_obs_ot = b["n_obs_ot"].to(device)

            hT, loss = model(times, time_ptr, X, obs_idx, delta_t, T, start_X,
                             n_obs_ot, return_path=False, get_loss=True)
            loss.backward()
            optimizer.step()
        train_time = time.time() - t

        # -------- evaluation --------
        t = time.time()
        batch = None
        with torch.no_grad():
            loss_val = 0
            num_obs = 0
            model.eval()  # set model in evaluation mode
            for i, b in enumerate(dl_val):
                if plot:
                    batch = b
                times = b["times"]
                time_ptr = b["time_ptr"]
                X = b["X"].to(device)
                start_X = b["start_X"].to(device)
                obs_idx = b["obs_idx"]
                n_obs_ot = b["n_obs_ot"].to(device)

                hT, c_loss = model(
                    times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
                    return_path=False, get_loss=True)


                loss_val += c_loss.cpu().numpy()
                num_obs += 1

            eval_time = time.time() - t
            loss_val = loss_val / num_obs
            train_loss = loss.detach().numpy()
            print("epoch {}, weight={:.5f}, train-loss={:.5f}, "
                  "optimal-eval-loss={:.5f}, eval-loss={:.5f}, ".format(
                model.epoch, model.weight, train_loss, opt_eval_loss, loss_val))
        metric_app.append([model.epoch, train_time, eval_time, train_loss,
                           loss_val, opt_eval_loss])

        # save model
        if model.epoch % save_every == 0:
            if plot:
                print('plotting ...')
                plot_save_path = '{}plots/'.format(model_path)
                plot_filename = 'epoch-{}'.format(model.epoch)
                plot_filename = plot_filename + '_path-{}.pdf'
                curr_opt_loss = plot_one_path_with_pred(
                    device, model, batch, stockmodel, delta_t, T,
                    path_to_plot=paths_to_plot, save_path=plot_save_path,
                    filename=plot_filename
                )
                print('optimal eval-loss (with current weight={:.5f}): '
                      '{:.5f}'.format(model.weight, curr_opt_loss))
            print('save model ...')
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            metric_app = []
            print('saved!')
        if loss_val < best_eval_loss:
            print('save new best model: last-best-loss: {:.5f}, '
                  'new-best-loss: {:.5f}, epoch: {}'.format(
                best_eval_loss, loss_val, model.epoch))
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            models.save_checkpoint(model, optimizer, model_path_save_best,
                                   model.epoch)
            metric_app = []
            best_eval_loss = loss_val
            print('saved!')

        model.epoch += 1
        model.weight_decay_step()



def compute_optimal_eval_loss(dl_val, stockmodel, delta_t, T):
    """
    compute optimal evaluation loss (with the true cond. exp.) on the
    test-dataset
    :param dl_val: torch.DataLoader, used for the validation dataset
    :param stockmodel: stock_model.StockModel instance
    :param delta_t: float, the time_delta
    :param T: float, the terminal time
    :return: float (optimal loss)
    """
    opt_loss = 0
    num_obs = 0
    for i, b in enumerate(dl_val):
        times = b["times"]
        time_ptr = b["time_ptr"]
        X = b["X"].detach().numpy()
        start_X = b["start_X"].detach().numpy()
        obs_idx = b["obs_idx"].detach().numpy()
        n_obs_ot = b["n_obs_ot"].detach().numpy()
        num_obs += 1
        opt_loss += stockmodel.get_optimal_loss(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot)
    return opt_loss/num_obs


def plot_one_path_with_pred(
        device, model, batch, stockmodel, delta_t, T,
        path_to_plot=(0,), save_path='', filename='plot_{}.pdf'
):
    """
    plot one path of the stockmodel together with optimal cond. exp. and its
    prediction by the model
    :param device: torch.device, to use for computations
    :param model: models.CODERNN instance
    :param batch: the batch from where to take the paths
    :param stockmodel: stock_model.StockModel instance, used to compute true
            cond. exp.
    :param delta_t: float
    :param T: float
    :param path_to_plot: list of ints, which paths to plot (i.e. which elements
            oof the batch)
    :param save_path: str, the path where to save the plot
    :param filename: str, the filename for the plot, should have one insertion
            possibility to put the path number
    :return: optimal loss
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    makedirs(save_path)

    times = batch["times"]
    time_ptr = batch["time_ptr"]
    X = batch["X"].to(device)
    start_X = batch["start_X"].to(device)
    obs_idx = batch["obs_idx"]
    n_obs_ot = batch["n_obs_ot"].to(device)
    true_X = batch["true_paths"]
    observed_dates = batch['observed_dates']
    path_t_true_X = np.linspace(0., T, int(T/delta_t)+1)
    
    model.eval()
    res = model.get_pred(times, time_ptr, X, obs_idx, delta_t, T,
                                  start_X)
    path_y_pred = res['pred'].detach().numpy()
    path_t_pred = res['pred_t']
    
    opt_loss, path_t_true, path_y_true = stockmodel.compute_cond_exp(
        times, time_ptr, X.detach().numpy(), obs_idx.detach().numpy(),
        delta_t, T, start_X.detach().numpy(), n_obs_ot.detach().numpy(),
        return_path=True, get_loss=True, weight=model.weight
    )

    for i in path_to_plot:
        # get the true_X at observed dates
        path_t_obs = [0.]
        path_X_obs = [true_X[i, :, 0]]
        for j, od in enumerate(observed_dates[i]):
            if od == 1:
                path_t_obs.append(path_t_true_X[j])
                path_X_obs.append(true_X[i, :, j])
        path_t_obs = np.array(path_t_obs)
        path_X_obs = np.array(path_X_obs)

        dim = true_X.shape[1]
        fig, axs = plt.subplots(dim)
        if dim == 1:
            axs = [axs]
        for j in range(dim):
            axs[j].plot(path_t_true_X, true_X[i, j, :], label='true path',
                        color=colors[0])
            axs[j].scatter(path_t_obs, path_X_obs[:, j], label='observed',
                           color=colors[0])
            axs[j].plot(path_t_pred, path_y_pred[:, i, j],
                        label='controlled ODE-RNN', color=colors[1])
            axs[j].plot(path_t_true, path_y_true[:, i, j],
                        label='true conditional expectation',
                        linestyle=':', color=colors[2])
        plt.legend()
        save = os.path.join(save_path, filename.format(i))
        plt.savefig(save)
        plt.close()
    
    return opt_loss







