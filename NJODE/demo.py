"""
author: Florian Krach & Calypso Herrera

run the demo version of the code:
    - load a pre-trained model & evaluate it
    - train a new model
"""


# =====================================================================================================================
import argparse
import numpy as np
import os, sys

sys.path.append("../")
try:
    from . import data_utils as data_utils
    from . import train as train
except Exception:
    import NJODE.data_utils as data_utils
    import NJODE.train as train




# =====================================================================================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Running controlled ODE-RNN")
    parser.add_argument(
        '--dataset', type=str,
        help="one of: 'BlackScholes', 'Heston', 'OrnsteinUhlenbeck'",
        default="BlackScholes")
    parser.add_argument(
        '--model_id', type=str,
        help="None or int of a pretrained model",
        default="None")
    parser.add_argument(
        '--epochs', type=int,
        help="int, number of epochs",
        default=200)
    args = parser.parse_args()
    model_id = args.model_id
    try:
        model_id = int(model_id)
    except Exception:
        model_id = None
    dataset = args.dataset
    save_every = 5
    epochs = args.epochs
    plot_only = False
    if model_id in [1,2,3]:
        print('use pretrained model ...')
        save_every = 1
        if model_id == 1:
            dataset = 'BlackScholes'
        if model_id == 2:
            dataset = 'Heston'
        if model_id == 3:
            dataset = 'OrnsteinUhlenbeck'
        plot_only = True


    if not os.path.exists(data_utils.training_data_path) or \
            not np.any([dataset in x for x in os.listdir(
                data_utils.training_data_path)]):
        print('no dataset exists for: {} -> generate dateset...'.format(dataset))
        dataset_dict = data_utils.hyperparam_default
        dataset_dict['nb_paths'] = 20000
        if model_id in [1, 2, 3]:
            dataset_dict['nb_paths'] = 100
        datasetpath, dataset_id = data_utils.create_dataset(
            stock_model_name=dataset, hyperparam_dict=dataset_dict)
        print('dataset stored as: {}'.format(datasetpath))

    ode_nn = ((50, 'tanh'), (50, 'tanh'))
    readout_nn = ((50, 'tanh'), (50, 'tanh'))
    enc_nn = ((50, 'tanh'), (50, 'tanh'))

    train.train(
        model_id=model_id, epochs=epochs, batch_size=100, save_every=save_every,
        learning_rate=0.001, test_size=0.2, seed=398,
        hidden_size=10, bias=True, dropout_rate=0.1,
        ode_nn=ode_nn, enc_nn=enc_nn,
        readout_nn=readout_nn, use_rnn=False,
        which_loss='standard', residual_enc_dec=True,
        solver="euler", weight=0.5, weight_decay=1.,
        dataset=dataset, dataset_id=None, paths_to_plot=(1, 2, 3, 4,),
        plot_only=plot_only)


