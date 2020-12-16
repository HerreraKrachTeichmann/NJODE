"""
author: Florian Krach & Calypso Herrera

code for aall additional things, like plotting, getting overviews etc.
"""


import pandas as pd
import numpy as np
import pdf2image
import imageio
import os, sys, time
import json
import socket
import matplotlib
import matplotlib.colors

from telegram_notifications import send_bot_message as SBM

if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True

SEND = False
if SERVER:
    SEND = True
    matplotlib.use('Agg')

sys.path.append("../")
try:
    from . import train as train
except Exception:
    import NJODE.train as train
import matplotlib.pyplot as plt



# ==============================================================================
def plot_loss_diff(
        path, filename, losses, xlab='epoch',
        ylab='$[\Psi(Y) - \Psi(\hat{X})]/\Psi(\hat{X})$',
        save_extras={}, fig_size=None,
):
    """
    function to plot the loss difference
    :param path: str, path where to save
    :param filename: str, filename to save
    :param losses: list, each element is a list with times (epochs), loss_diff,
            and name for the legend
    :param xlab: str
    :param ylab: str,
    :param save_extras: kwargs for saving
    :param fig_size: None or list of 2 ints
    """
    if fig_size is not None:
        f = plt.figure(figsize=fig_size)
    else:
        f = plt.figure()
    for tup in losses:
        t, loss_diff, name = tup
        plt.plot(t, loss_diff, label=name)
    plt.legend()
    if xlab is not None:
        plt.xlabel(xlab)
    if ylab is not None:
        plt.ylabel(ylab)
    plt.savefig(os.path.join(path, filename), **save_extras)
    plt.close()


def plot_losses(files, names, time_col='epoch', 
                col1='eval_loss', col2='optimal_eval_loss', 
                relative_error=True,
                filename='plot.pdf', path='./',
                save_extras={'bbox_inches':'tight', 'pad_inches': 0.01},
                **kwargs):
    """
    function to plot the loss vs. epoch of trained models
    :param files: list, each element is one path for a loss-file
    :param names: list, the names for the legend
    :param time_col: str, the name of the "time" (epoch) column in the files
    :param col1: str, the name of the eval-loss column
    :param col2: str, the name of the optimal eval loss column
    :param relative_error: bool, whether to plot a relative or absolute diff
    :param filename: str, the output filename
    :param path: str, output path
    :param save_extras: dict, additional kwargs for saving
    :param kwargs: additional key-word arguments, passed on
    """
    losses = []
    for file, name in zip(files, names):
        df = pd.read_csv(file, index_col=0)
        t = df[time_col].values
        loss = df[col1].values - df[col2].values
        if relative_error:
            loss = loss / df[col2].values
        losses.append([t, loss, name])
    plot_loss_diff(path, filename, losses, save_extras=save_extras, **kwargs)


def generate_training_progress_gif(model_id, which_path=1):
    """
    generate a gif of the training progress for a model id and path, i.e. a gif
    of the evolution of the path over the training epochs
    saved into plot dir of the model
    :param model_id: int
    :param which_path: int
    """
    path = train.saved_models_path
    path = os.path.join(path, 'id-{}/plots/'.format(model_id))
    filenames = []
    for f in sorted(os.listdir(path)):
        if 'path-{}.pdf'.format(which_path) in f:
            print('converting {} to png'.format(f))
            im = pdf2image.convert_from_path(os.path.join(path, f), 100)
            savepath = os.path.join(path, f[:-3]+'png')
            for i in im:
                i.save(savepath, 'PNG')
            filenames.append(savepath)
    images = []
    for filename in sorted(filenames,
                           key=lambda s: int(s.split('epoch-')[1].split('_')[0])):
        print(filename)
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(path,'training-progress-path-{}.gif'.format(
        which_path)), images, duration=0.5)


def plot_convergence_study(
        path=train.saved_models_path, ids_from=None,
        ids_to=None, x_axis="training_size",
        x_log=False, y_log=False,
        save_path="{}plots/".format(train.data_path),
        save_extras={'bbox_inches':'tight', 'pad_inches': 0.01}
):
    """
    function to plot the convergence study when network size and number of
    samples are varied and multiple "identical" models are trained for each
    combiation
    :param path: str, path where models are saved
    :param ids_from: None or int, which ids to consider start point
    :param ids_to: None or int, which ids to consider end point
    :param x_axis: str, one of {"training_size", "network_size"}, which of the
            two is on the x-axis
    :param x_log: bool, whether x-axis is logarithmic
    :param y_log: bool, whether y-axis is logarithmic
    :param save_path: str
    :param save_extras: dict, extra arguments for saving the plots
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    filename = "{}model_overview.csv".format(path)
    df = pd.read_csv(filename, index_col=0)
    if ids_from:
        df = df.loc[df["id"] >= ids_from]
    if ids_to:
        df = df.loc[df["id"] <= ids_to]

    # extract network size and number training samples
    df["network_size"] = None
    df["training_size"] = None
    for i in df.index:
        desc = df.loc[i, "description"]
        param_dict = json.loads(desc)
        training_size = param_dict["training_size"]
        network_size = param_dict["enc_nn"][0][0]
        df.loc[i, ["network_size",  "training_size"]] = \
            [network_size, training_size]

    # get sets of network size and training size
    n_sizes = sorted(list(set(df["network_size"].values)))
    t_sizes = sorted(list(set(df["training_size"].values)))
    if x_axis == "training_size":
        x_axis_params = t_sizes
        other_param_name = "network_size"
        other_params = n_sizes
    else:
        x_axis = "network_size"
        x_axis_params = n_sizes
        other_param_name = "training_size"
        other_params = t_sizes

    # get means and stds
    means = []
    stds = []
    for val2 in other_params:
        _means = []
        _stds = []
        for val1 in x_axis_params:
            current_losses = []
            ids = df.loc[(df[x_axis] == val1) & (df[other_param_name] == val2),
                         "id"]
            for id in ids:
                file_n = "{}id-{}/metric_id-{}.csv".format(path, id, id)
                df_metric = pd.read_csv(file_n, index_col=0)
                current_losses.append(np.min(df_metric["evaluation_mean_diff"]))
            _means.append(np.mean(current_losses))
            _stds.append(np.std(current_losses))
        means.append(_means)
        stds.append(_stds)

    # plotting
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    for i, args in enumerate(zip(means, stds, other_params)):
        mean, std, val2 = args
        ax.errorbar(x_axis_params, mean, yerr=std,
                    label="{}={}".format(other_param_name, val2),
                    ecolor="black", capsize=4, capthick=1, marker=".",
                    color=colors[i])
    plt.xlabel(x_axis)
    plt.ylabel("eval metric")
    plt.legend()
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = "{}convergence_{}.png".format(save_path, x_axis)
    plt.savefig(save_file, **save_extras)

    if SEND:
        files_to_send = [save_file]
        SBM.send_notification(
            text_for_files='convergence plot',
            files=files_to_send
        )


def get_training_overview(
        path=train.saved_models_path, ids_from=None,
        ids_to=None,
        params_extract_desc=('network_size', 'training_size', 'dataset',
                             'hidden_size'),
        val_test_params_extract=(("max", "epoch", "epoch", "epochs_trained"),
                                 ("min", "evaluation_mean_diff",
                                  "evaluation_mean_diff", "eval_metric_min"),
                                 ("last", "evaluation_mean_diff",
                                  "evaluation_mean_diff", "eval_metric_last"),
                                 ("average", "evaluation_mean_diff",
                                  "evaluation_mean_diff", "eval_metric_average")
                                 ),
        early_stop_after_epoch=0,
        save_file=None,
):
    """
    function to get the important metrics and hyper-params for each model in the
    models_overview.csv file
    :param path: str, where the saved models are
    :param ids_from: None or int, which model ids to consider start point
    :param ids_to: None or int, which model ids to consider end point
    :param params_extract_desc: list of str, names of params to extract from the
            model description dict, special:
                - network_size: gets size of first layer of enc network
                - activation_function_x: gets the activation function of layer x
                    of enc network
    :param val_test_params_extract: None or list of list with 4 string elements:
                0. "min" or "max" or "last" or "average"
                1. col_name where to look for min/max (validation), or where to
                    get last value or average
                2. if 0. is min/max: col_name where to find value in epoch where
                                     1. is min/max (test)
                   if 0. is last/average: not used
                3. name for this output column in overview file
    :param early_stop_after_epoch: int, epoch after which early stopping is
            allowed (i.e. all epochs until there are not considered)
    :param save_file:
    :return:
    """
    filename = "{}model_overview.csv".format(path)
    df = pd.read_csv(filename, index_col=0)
    if ids_from:
        df = df.loc[df["id"] >= ids_from]
    if ids_to:
        df = df.loc[df["id"] <= ids_to]

    # extract wanted information
    for param in params_extract_desc:
        df[param] = None

    if val_test_params_extract:
        for l in val_test_params_extract:
            df[l[3]] = None

    for i in df.index:
        desc = df.loc[i, "description"]
        param_dict = json.loads(desc)

        values = []
        for param in params_extract_desc:
            try:
                if param == 'network_size':
                    v = param_dict["enc_nn"][0][0]
                elif 'activation_function' in param:
                    numb = int(param.split('_')[-1])
                    v = param_dict["enc_nn"][numb-1][1]
                else:
                    v = param_dict[param]
                values.append(v)
            except Exception:
                values.append(None)
        df.loc[i, params_extract_desc] = values

        id = df.loc[i, "id"]
        file_n = "{}id-{}/metric_id-{}.csv".format(path, id, id)
        df_metric = pd.read_csv(file_n, index_col=0)
        if early_stop_after_epoch:
            df_metric = df_metric.loc[df_metric['epoch']>early_stop_after_epoch]

        if val_test_params_extract:
            for l in val_test_params_extract:
                if l[0] == 'max':
                    f = np.nanmax
                elif l[0] == 'min':
                    f = np.nanmin

                if l[0] in ['min', 'max']:
                    try:
                        ind = (df_metric.loc[df_metric[l[1]] ==
                                             f(df_metric[l[1]])]).index[0]
                        df.loc[i, l[3]] = df_metric.loc[ind, l[2]]
                    except Exception:
                        pass
                elif l[0] == 'last':
                    df.loc[i, l[3]] = df_metric[l[1]].values[-1]
                elif l[0] == 'average':
                    df.loc[i, l[3]] = np.nanmean(df_metric[l[1]])

    # save
    if save_file is not False:
        if save_file is None:
            save_file = "{}model_overview-training_results.csv".format(path)
        df.to_csv(save_file)

    if SEND and save_file is not False:
        files_to_send = [save_file]
        SBM.send_notification(
            text=None,
            text_for_files='training overview',
            files=files_to_send
        )

    return df


def plot_paths_from_checkpoint(
        model_ids=(1,), which='best', paths_to_plot=(0,), **options
):
    """
    function to plot paths (using plot_one_path_with_pred) from a saved model
    checkpoint
    :param model_ids: list of int, the ids of the models to load and plot
    :param which: one of {'best', 'last', 'both'}, which checkpoint to load
    :param path_to_plot: list of int, see train.train.py
    :param options: feed directly to train
    :return:
    """
    model_overview_file_name = '{}model_overview.csv'.format(
        train.saved_models_path)
    if not os.path.exists(model_overview_file_name):
        print("No saved model_overview.csv file")
        return 1
    else:
        df_overview = pd.read_csv(model_overview_file_name, index_col=0)

    for model_id in model_ids:
        if model_id not in df_overview['id'].values:
            print("model_id={} does not exist yet -> skip".format(model_id))
        else:
            desc = (df_overview['description'].loc[
                df_overview['id'] == model_id]).values[0]
            params_dict = json.loads(desc)
            params_dict['model_id'] = model_id
            params_dict['resume_training'] = True
            params_dict['plot_only'] = True
            params_dict['paths_to_plot'] = paths_to_plot
            params_dict['parallel'] = True
            for key in options:
                params_dict[key] = options[key]

            if which in ['best', 'both']:
                params_dict['load_best'] = True
                train.train(**params_dict)
            if which in ['last', 'both']:
                params_dict['load_best'] = False
                train.train(**params_dict)


def plot_loss_and_metric(
        model_ids=(1,),
        save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
        file_name="loss_and_metric-id{}.pdf",
        time_col='epoch',
        cols=('train_loss', 'eval_loss', 'evaluation_mean_diff'),
        names=('train_loss', 'eval_loss', 'eval_metric')
):
    """
    function to plot the losses and metric in one plot with subplots to see
    their joint evolution
    :param model_ids: list of int
    :param save_extras: dict
    :param file_name: str including "{}", name of saved file
    :param time_col: str, usually 'epoch'
    :param cols: list of str, the column names to plot
    :param names: None or list of str, names of the y-labels, None: use cols
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    if names is None:
        names = cols

    for model_id in model_ids:
        path = os.path.join(
            train.saved_models_path, "id-{}".format(model_id),
            "metric_id-{}.csv".format(model_id)
        )
        df = pd.read_csv(path)
        t = df[time_col]
        n = len(cols)
        fig, axes = plt.subplots(n)
        for i, col in enumerate(cols):
            axes[i].plot(t, df[col].values, color=colors[i])
            axes[i].set(ylabel=names[i])
        axes[-1].set(xlabel=time_col)

        save_path = os.path.join(
            train.saved_models_path, "id-{}".format(model_id),
            file_name.format(model_id)
        )
        plt.savefig(save_path, **save_extras)
        plt.close(fig)

        if SEND:
            SBM.send_notification(
                text=None, files=[save_path],
                text_for_files="loss and metric plot - id={}".format(model_id)
            )


def get_cross_validation(
        params_extract_desc=('dataset', 'network_size', 'dropout_rate',
                             'hidden_size', 'activation_function_1'),
        val_test_params_extract=(("min", "eval_metric",
                                  "test_metric", "test_metric_evaluation_min"),
                                 ("min", "eval_metric",
                                  "eval_metric", "eval_metric_min")
                                 ),
        target_col=('eval_metric_min', 'test_metric_evaluation_min'),
        early_stop_after_epoch=0,
        param_combinations=({'network_size': 50,
                             'activation_function_1': 'tanh',
                             'dropout_rate': 0.1,
                             'hidden_size': 10,
                             'dataset': 'climate'},
                            {'network_size': 200,
                             'activation_function_1': 'tanh',
                             'dropout_rate': 0.1,
                             'hidden_size': 10,
                             'dataset': 'climate'},
                            {'network_size': 400,
                             'activation_function_1': 'tanh',
                             'dropout_rate': 0.1,
                             'hidden_size': 50,
                             'dataset': 'climate'},
                            {'network_size': 50,
                             'activation_function_1': 'relu',
                             'dropout_rate': 0.2,
                             'hidden_size': 50,
                             'dataset': 'climate'},
                            {'network_size': 100,
                             'activation_function_1': 'relu',
                             'dropout_rate': 0.2,
                             'hidden_size': 50,
                             'dataset': 'climate'},
                            {'network_size': 400,
                             'activation_function_1': 'relu',
                             'dropout_rate': 0.2,
                             'hidden_size': 10,
                             'dataset': 'climate'}),
        save_path="{}climate_cross_val.csv".format(train.saved_models_path),
        path=train.saved_models_path
):
    """
    function to get the cross validation of the climate dataset
    :param params_extract_desc: list of str, see get_training_overview()
    :param val_test_params_extract: lst of list, see get_training_overview()
    :param target_col: list of str, column (generated by get_training_overview()
            to perform cross-validation on
    :param early_stop_after_epoch: int, see get_training_overview()
    :param param_combinations: list of dict, each dict definnes one combination
            of params, which are used as one sample for the cross-validation
            (mean is then taken over all other params that are not specfied,
            where the specifiied params are the same)
    :param save_path: str, where to save the output file
    :param path: str, path where models are saved
    :return: pd.DataFrame with cross val mean and std
    """
    df = get_training_overview(
        path=path,
        params_extract_desc=params_extract_desc,
        val_test_params_extract=val_test_params_extract,
        early_stop_after_epoch=early_stop_after_epoch,
        save_file=False
    )

    data = []
    for pc in param_combinations:
        df_ = df.copy()
        name = json.dumps(pc, sort_keys=True)
        data_ = [name]
        for key in pc:
            df_ = df_.loc[df_[key] == pc[key]]
        for tc in target_col:
            vals = df_[tc]
            data_ += [np.mean(vals), np.std(vals)]
        data.append(data_)

    columns = ['param_combination']
    for tc in target_col:
        columns += ['mean_{}'.format(tc), 'std_{}'.format(tc)]
    df_out = pd.DataFrame(
        data=data, columns=columns)
    df_out.to_csv(save_path)

    if SEND:
        SBM.send_notification(
            text=None, files=[save_path],
            text_for_files="cross validation"
        )

    return df_out


if __name__ == '__main__':
    # plot_losses(['/Users/Flo/Code/GitRepos/ControlledODERNN/data/saved_models/id-1/metric_id-1.csv',
    #              '/Users/Flo/Code/GitRepos/ControlledODERNN/data/saved_models/id-2/metric_id-2.csv',
    #              '/Users/Flo/Code/GitRepos/ControlledODERNN/data/saved_models/id-3/metric_id-3.csv'],
    #             ['Black-Scholes', 'Heston', 'Ornstein-Uhlenbeck'],
    #             filename='relative_difference.pdf', path='/Users/Flo/Desktop/',
    #             ylab='relative loss', fig_size=(8,4))


    # generate_training_progress_gif(6, which_path=0)


    # plot_convergence_study(x_axis="training_size", x_log=True, y_log=True)
    # plot_convergence_study(x_axis="network_size", x_log=True, y_log=True)


    # get_training_overview(
    #     params_extract_desc=('dataset', 'dataset_id', "other_model",
    #                          'network_size', 'training_size',
    #                          'hidden_size', "GRU_ODE_Bayes-mixing",
    #                          "GRU_ODE_Bayes-logvar", "GRU_ODE_Bayes-impute",
    #                          "GRU_ODE_Bayes-mixing"),
    #     val_test_params_extract=(("max", "epoch", "epoch", "epochs_trained"),
    #                              ("min", "evaluation_mean_diff",
    #                               "evaluation_mean_diff", "eval_metric_min"),
    #                              ("last", "evaluation_mean_diff",
    #                               "evaluation_mean_diff", "eval_metric_last"),
    #                              ("average", "evaluation_mean_diff",
    #                               "evaluation_mean_diff", "eval_metric_average")
    #                              )
    # )


    # for model_id in (2, 16, 12, 49, 50, 51):
    #     plot_paths_from_checkpoint(
    #         model_ids=(model_id,), which='both',
    #         paths_to_plot=(0, 1, 2, 3, 4)
    #     )
    #     time.sleep(60)

    # plot_loss_and_metric(model_ids=(2, 16, 12, 49, 50, 51))


    # # ------------ validation of climate training -------------
    # path = '{}saved_models_climate2/'.format(train.data_path)
    # get_training_overview(
    #     path=path,
    #     params_extract_desc=('dataset', 'network_size', 'dropout_rate',
    #                          'hidden_size', 'data_index'),
    #     val_test_params_extract=(("max", "epoch", "epoch", "epochs_trained"),
    #                              ("min", "eval_metric",
    #                               "eval_metric", "eval_metric_min"),
    #                              ("min", "test_metric",
    #                               "test_metric", "test_metric_min"),
    #                              ("min", "eval_metric",
    #                               "test_metric", "test_metric_evaluation_min"),
    #                              ("min", "eval_loss",
    #                               "test_metric", "test_metric_eval_loss_min"),
    #                              )
    # )
    # get_cross_validation(
    #     path=path, save_path='{}cross_val.csv'.format(path),
    #     early_stop_after_epoch=100,
    #     param_combinations=({'hidden_size': 10},
    #                         {'hidden_size': 50})
    # )



    # plot_paths_from_checkpoint(
    #     model_ids=(2,), which='last', paths_to_plot=(0, 1, 2, 3),
    #     ylabels=('X', 'v'),
    #     save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
    # )



    # # ------------ validation of physionet training -------------
    # get_training_overview(
    #     path='{}saved_models_physionet_comparison/'.format(train.data_path),
    #     params_extract_desc=('dataset', 'network_size', 'dropout_rate',
    #                          'hidden_size', 'data_index'),
    #     val_test_params_extract=(("max", "epoch", "epoch", "epochs_trained"),
    #                              ("min", "eval_metric",
    #                               "eval_metric", "eval_metric_min"),
    #                              ("min", "eval_metric_2",
    #                               "eval_metric_2", "eval_metric_2_min"),
    #                              )
    # )
    #
    # get_cross_validation(
    #     path='{}saved_models_physionet_comparison/'.format(train.data_path),
    #     save_path='{}saved_models_physionet_comparison/'
    #               'cross_val.csv'.format(train.data_path),
    #     param_combinations=({'network_size': 50},
    #                         {'network_size': 200}),
    #     val_test_params_extract=(("max", "epoch", "epoch", "epochs_trained"),
    #                              ("min", "eval_metric",
    #                               "eval_metric", "eval_metric_min"),
    #                              ("min", "eval_metric_2",
    #                               "eval_metric_2", "eval_metric_2_min"),
    #                              ("last", "eval_metric_2",
    #                               "eval_metric_2", "eval_metric_2_last"),
    #                              ("min", "train_loss",
    #                               "eval_metric_2", "eval_metric_2_eval_min"),
    #                              ),
    #     target_col=('eval_metric_min', 'eval_metric_2_min',
    #                 'eval_metric_2_last', 'eval_metric_2_eval_min')
    # )


    pass



