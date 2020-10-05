"""
author: Florian Krach & Calypso Herrera

implementation of the training (and evaluation) of NJ-ODE
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
    N_CPUS = 2
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
    from ..GRU_ODE_Bayes import models_gru_ode_bayes as models_gru_ode_bayes
except Exception:
    import NJODE.models as models
    import NJODE.data_utils as data_utils
    import GRU_ODE_Bayes.models_gru_ode_bayes as models_gru_ode_bayes
import matplotlib.pyplot as plt


# =====================================================================================================================
# Global variables
data_path = data_utils.data_path
saved_models_path = '{}saved_models/'.format(data_path)

METR_COLUMNS = ['epoch', 'train_time', 'eval_time', 'train_loss', 'eval_loss',
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
        solver="euler", weight=1., weight_decay=0.999,
        dataset='BlackScholes', dataset_id=None, plot=True, paths_to_plot=(0,),
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
    :param test_size: float in (0,1), the percentage of samples to use for the
            test set (here there exists only a test set, since there is no extra
            evaluation)
    :param seed: int, seed for the random splitting of the dataset into train
            and test
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
    :param dataset: str, which dataset to use, supported: {'BlackScholes',
            'Heston', 'OrnsteinUhlenbeck'}. The corresponding dataset already
            needs to exist (create it first using data_utils.create_dataset)
    :param dataset_id: int or None, the id of the dataset to be use, if None,
            the latest generated dataset of the given name will be used
    :param plot: bool, whethere to plot
    :param paths_to_plot: list of ints, which paths of the test-set should be
            plotted
    :param options: kwargs, used keywords:
            'func_appl_X'   list of functions (as str, see data_utils)
                            to apply to X
            'plot_variance' bool, whether to plot also variance
            'std_factor'    float, the factor by which the std is multiplied
            'parallel'      bool, used by parallel_train.parallel_training
            'resume_training'   bool, used by parallel_train.parallel_training
            'plot_only'     bool, whether the model is used only to plot after
                            initiating or loading (i.e. no training) and exit
                            afterwards (used by demo)
            'which_loss'    'standard' or 'easy', used by models.NJODE
            'residual_enc_dec'  bool, whether resNNs are used for encoder and
                                readout NN, used by models.NJODE, default True
            'training_size' int, if given and smaller than
                            dataset_size*(1-test_size), then this is the umber
                            of samples used for the training set (randomly
                            selected out of original training set)
            'evaluate'      bool, whether to evaluate the model in the test set
                            (i.e. not only compute the eval_loss, but also
                            compute the mean difference between the true and the
                            predicted paths comparing at each time point)
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

    # load dataset-metadata
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
    # -- get subset of training samples if wanted
    if 'training_size' in options:
        train_set_size = options['training_size']
        if train_set_size < len(train_idx):
            train_idx = np.random.choice(
                train_idx, train_set_size, replace=False
            )
    data_train = data_utils.IrregularDataset(
        model_name=dataset, time_id=dataset_id, idx=train_idx)
    data_val = data_utils.IrregularDataset(
        model_name=dataset, time_id=dataset_id, idx=val_idx)
    
    # get data-loader for training
    if 'func_appl_X' in options:
        functions = options['func_appl_X']
        collate_fn, mult = data_utils.CustomCollateFnGen(functions)
        input_size = input_size * mult
        output_size = output_size * mult
    else:
        functions = None
        mult = 1
        collate_fn = data_utils.custom_collate_fn

    dl = DataLoader(
        dataset=data_train, collate_fn=collate_fn,
        shuffle=True, batch_size=batch_size, num_workers=N_DATASET_WORKERS)
    dl_val = DataLoader(
        dataset=data_val, collate_fn=collate_fn,
        shuffle=False, batch_size=len(data_val), num_workers=N_DATASET_WORKERS)
    
    # get additional plotting information
    plot_variance = False
    std_factor = 1
    if functions is not None and mult > 1:
        if 'plot_variance' in options:
            plot_variance = options['plot_variance']
        if 'std_factor' in options:
            std_factor = options['std_factor']

    # get optimal eval loss
    #    TODO: this is not correct if other functions are applied to X
    stockmodel = data_utils._STOCK_MODELS[
        dataset_metadata['model_name']](**dataset_metadata)
    opt_eval_loss = compute_optimal_eval_loss(
        dl_val, stockmodel, delta_t, T)
    initial_print += '\noptimal eval loss (achieved by true cond exp): ' \
                     '{:.5f}'.format(opt_eval_loss)
    if 'other_model' in options:
        opt_eval_loss = np.nan

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
    if ('parallel' in options and options['parallel'] is False) or \
            ('parallel' not in options):
        model_overview_file_name = '{}model_overview.csv'.format(
            saved_models_path
        )
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
            initial_print += '\nnew model_id={}'.format(model_id)
            df_ov_app = pd.DataFrame([[model_id, desc]],
                                     columns=['id', 'description'])
            df_overview = pd.concat([df_overview, df_ov_app], ignore_index=True)
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
    plot_save_path = '{}plots/'.format(model_path)
    if 'save_extras' in options:
        save_extras = options['save_extras']
    else:
        save_extras = {}

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
        raise ValueError("Invalid argument for (option) parameter 'other_model'."
                   "Please check docstring for correct use.")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=0.0005)

    # load saved model if wanted/possible
    best_eval_loss = np.infty
    if 'evaluate' in options and options['evaluate']:
        metr_columns = METR_COLUMNS + ['evaluation_mean_diff']
    else:
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
            best_eval_loss = np.min(df_metric['eval_loss'].values)
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

    # ---------- plot only option for demo ------------
    if 'plot_only' in options and options['plot_only']:
        for i, b in enumerate(dl_val):
                batch = b
        model.epoch -= 1
        initial_print += '\nplotting ...'
        plot_filename = 'demo-plot_epoch-{}'.format(model.epoch)
        plot_filename = plot_filename + '_path-{}.pdf'
        curr_opt_loss = plot_one_path_with_pred(
            device, model, batch, stockmodel, delta_t, T,
            path_to_plot=paths_to_plot, save_path=plot_save_path,
            filename=plot_filename, plot_variance=plot_variance,
            functions=functions, std_factor=std_factor,
            model_name=model_name, save_extras=save_extras
        )
        # if SEND:
        #     files_to_send = []
        #     caption = "{} - id={}".format(model_name, model_id)
        #     for i in paths_to_plot:
        #         files_to_send.append(
        #             os.path.join(plot_save_path, plot_filename.format(i)))
        #     SBM.send_notification(
        #         text='finished plot-only: {}, id={}\n\n{}'.format(
        #             model_name, model_id, desc),
        #         files=files_to_send,
        #         text_for_files=caption
        #     )
        initial_print += '\noptimal eval-loss (with current weight={:.5f}): ' \
                         '{:.5f}'.format(model.weight, curr_opt_loss)
        print(initial_print)
        return 0

    # ---------------- TRAINING ----------------
    skip_training = True
    if model.epoch <= epochs:
        skip_training = False

        # # send notification
        # if SEND:
        #     SBM.send_notification(
        #         text='start training controlled ODE-RNN id={}'.format(model_id)
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
            start_X = b["start_X"].to(device)
            obs_idx = b["obs_idx"]
            # n_obs_ot = b["n_obs_ot"].to(device)
            #   n_obs_ot is sommetimes wrong in dataset -> but should not make a
            #   difference. however, as below it is coorrect.
            b_size = start_X.size()[0]
            unique_idx, counts = np.unique(
                obs_idx.detach().numpy(), return_counts=True)
            n_obs_ot = np.zeros((b_size))
            n_obs_ot[unique_idx] = counts
            n_obs_ot = n_obs_ot.astype(np.int)
            n_obs_ot = torch.tensor(n_obs_ot).to(device)

            if 'other_model' not in options:
                hT, loss = model(
                    times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
                    return_path=False, get_loss=True
                )
            elif options['other_model'] == "GRU_ODE_Bayes":
                M = torch.ones_like(X)
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
        batch = None
        with torch.no_grad():
            loss_val = 0
            num_obs = 0
            eval_msd = 0
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

                if 'other_model' not in options:
                    hT, c_loss = model(
                        times, time_ptr, X, obs_idx, delta_t, T, start_X,
                        n_obs_ot, return_path=False, get_loss=True
                    )
                elif options['other_model'] == "GRU_ODE_Bayes":
                    M = torch.ones_like(X)
                    hT, c_loss, _, _ = model(
                        times, time_ptr, X, M, obs_idx, delta_t, T, start_X,
                        return_path=False, smoother=False
                    )
                else:
                    raise ValueError

                loss_val += c_loss.detach().numpy()
                num_obs += 1

                # mean squared difference evaluation
                if 'evaluate' in options and options['evaluate']:
                    _eval_msd = model.evaluate(
                        times, time_ptr, X, obs_idx, delta_t, T, start_X,
                        n_obs_ot, stockmodel, return_paths=False)
                    eval_msd += _eval_msd

            eval_time = time.time() - t
            loss_val = loss_val / num_obs
            eval_msd = eval_msd / num_obs
            train_loss = loss.detach().numpy()
            print("epoch {}, weight={:.5f}, train-loss={:.5f}, "
                  "optimal-eval-loss={:.5f}, eval-loss={:.5f}, ".format(
                model.epoch, model.weight, train_loss, opt_eval_loss, loss_val))
        if 'evaluate' in options and options['evaluate']:
            metric_app.append([model.epoch, train_time, eval_time, train_loss,
                              loss_val, opt_eval_loss, eval_msd])
            print("evaluation mean square difference={:.5f}".format(
                eval_msd))
        else:
            metric_app.append([model.epoch, train_time, eval_time, train_loss,
                               loss_val, opt_eval_loss])

        # save model
        if model.epoch % save_every == 0:
            if plot:
                print('plotting ...')
                plot_filename = 'epoch-{}'.format(model.epoch)
                plot_filename = plot_filename + '_path-{}.pdf'
                curr_opt_loss = plot_one_path_with_pred(
                    device, model, batch, stockmodel, delta_t, T,
                    path_to_plot=paths_to_plot, save_path=plot_save_path,
                    filename=plot_filename, plot_variance=plot_variance,
                    functions=functions, std_factor=std_factor,
                    model_name=model_name, save_extras=save_extras
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

    # # send notification
    # if SEND and not skip_training:
    #     files_to_send = [model_metric_file]
    #     caption = "{} - id={}".format(model_name, model_id)
    #     if plot:
    #         for i in paths_to_plot:
    #             files_to_send.append(
    #                 os.path.join(plot_save_path, plot_filename.format(i)))
    #     SBM.send_notification(
    #         text='finished training: {}, id={}\n\n{}'.format(
    #             model_name, model_id, desc),
    #         files=files_to_send,
    #         text_for_files=caption
    #     )

    # delete model & free memory
    del model, dl, dl_val, data_train, data_val
    gc.collect()

    return 0


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
        path_to_plot=(0,), save_path='', filename='plot_{}.pdf',
        plot_variance=False, functions=None, std_factor=1,
        model_name=None,
        save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01}
):
    """
    plot one path of the stockmodel together with optimal cond. exp. and its
    prediction by the model
    :param device: torch.device, to use for computations
    :param model: models.NJODE instance
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
    :param plot_variance: bool, whether to plot the variance, if supported by
            functions (i.e. square has to be applied)
    :param functions: list of functions (as str), the functions applied to X
    :param std_factor: float, the factor by which std is multiplied
    :param model_name: str or None, name used for model in plots
    :param save_extras: dict with extra options for saving plot
    :return: optimal loss
    """
    if model_name is None:
        model_name = 'our model'

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    std_color = list(matplotlib.colors.to_rgb(colors[1])) + [0.5]

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
    
    # get variance path
    if plot_variance and (functions is not None) and ('power-2' in functions):
        which = np.argmax(np.array(functions) == 'power-2')
        dim = true_X.shape[1]
        y2 = path_y_pred[:, :, (dim*which):(dim*(which+1))]
        path_var_pred = y2 - np.power(path_y_pred[:, :, 0:dim], 2)
        if np.any(path_var_pred < 0):
            print('WARNING: some predicted cond. variances below 0 -> clip')
            path_var_pred = np.maximum(0, path_var_pred)
        path_std_pred = np.sqrt(path_var_pred)
    else:
        plot_variance = False

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
                        label=model_name, color=colors[1])
            if plot_variance:
                # axs[j].plot(
                #     path_t_pred,
                #     path_y_pred[:, i, j] + std_factor*path_std_pred[:, i, j],
                #     color=colors[1])
                # axs[j].plot(
                #     path_t_pred,
                #     path_y_pred[:, i, j] - std_factor*path_std_pred[:, i, j],
                #     color=colors[1])
                axs[j].fill_between(
                    path_t_pred,
                    path_y_pred[:, i, j] - std_factor * path_std_pred[:, i, j],
                    path_y_pred[:, i, j] + std_factor * path_std_pred[:, i, j],
                    color=std_color)
            axs[j].plot(path_t_true, path_y_true[:, i, j],
                        label='true conditional expectation',
                        linestyle=':', color=colors[2])

        plt.legend()
        plt.xlabel('$t$')
        save = os.path.join(save_path, filename.format(i))
        plt.savefig(save, **save_extras)
        plt.close()
    
    return opt_loss



if __name__ == '__main__':

    dataset_id = None
    dataset = "BlackScholes"
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
          dataset=dataset, dataset_id=dataset_id, paths_to_plot=(0,1,2,3,4,),
          evaluate=True)

    # train GRU-ODE-Bayes
    train(model_id=None, epochs=100, batch_size=100, save_every=1,
          learning_rate=0.001, test_size=0.2, seed=398,
          hidden_size=50, bias=True, dropout_rate=0.1,
          ode_nn=None, enc_nn=None,
          readout_nn=None, use_rnn=False,
          residual_enc_dec=True,
          solver="euler", weight=0.5, weight_decay=1.,
          dataset=dataset, dataset_id=dataset_id,
          paths_to_plot=(0, 1, 2, 3, 4,),
          evaluate=True, other_model='GRU_ODE_Bayes',
          **{'GRU_ODE_Bayes-impute': True, 'GRU_ODE_Bayes-logvar': True,
             'GRU_ODE_Bayes-mixing': 0.0001}
          )




