"""
authors: Florian Krach & Calypso Herrera

implementation of the model for NJ-ODE
"""

# =====================================================================================================================
import torch
import numpy as np
import os, sys
import socket

sys.path.append("../")
try:
    import NJODE.data_utils as data_utils
except Exception:
    from . import data_utils


# =====================================================================================================================
def init_weights(m, bias=0.0):
    # TODO: GRU-ODE-Bayes used bias=0.05
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(bias)


def save_checkpoint(model, optimizer, path, epoch):
    """
    save a trained torch model and the used optimizer at the given path, s.t.
    training can be resumed at the exact same point
    :param model: a torch model, e.g. instance of NJODE
    :param optimizer: a torch optimizer
    :param path: str, the path where to save the model
    :param epoch: int, the current epoch
    """
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, 'checkpt.tar')
    torch.save({'epoch': epoch,
                'weight': model.weight,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               filename)


def get_ckpt_model(ckpt_path, model, optimizer, device):
    """
    load a saved torch model and its optimizer, inplace
    :param ckpt_path: str, path where the model is saved
    :param model: torch model instance, of which the weights etc. should be
            reloaded
    :param optimizer: torch optimizer, which should be loaded
    :param device: the device to which the model should be loaded
    """
    ckpt_path = os.path.join(ckpt_path, 'checkpt.tar')
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    checkpt = torch.load(ckpt_path)
    state_dict = checkpt['model_state_dict']
    optimizer.load_state_dict(checkpt['optimizer_state_dict'])
    model.load_state_dict(state_dict)
    model.epoch = checkpt['epoch']
    model.weight = checkpt['weight']
    model.to(device)
    
    
    
def compute_loss(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
                 weight=0.5, M_obs=None):
    """
    loss function from the paper
    :param X_obs: torch.tensor, the true X values at the observations
    :param Y_obs: torch.tensor, the predicted values at the observation
    :param Y_obs_bj: torch.tensor, the predicted values before the jump at the
            observation
    :param n_obs_ot: torch.tensor, the number of observations over the entire
            time-line for each element of the batch
    :param batch_size: int or float
    :param eps: float, a small constant which is added before taking torch.sqrt
            s.t. the sqrt never becomes zero (which would yield NaNs for the
            gradient)
    :param weight: float in [0,1], weighting of the two parts of the loss
            function,
                0.5: standard loss as described in paper
                (0.5, 1): more weight to be correct after the jump, can be
                theoretically justified similar to standard loss
                1: only the value after the jump is trained
    :param M_obs: None or torch.tensor, if not None: same size as X_obs with
            0  and 1 entries, telling which coordinates were observed
    :return: torch.tensor (with the loss, reduced to 1 dim)
    """
    if M_obs is None:
        inner = (2*weight * torch.sqrt(torch.sum((X_obs - Y_obs) ** 2, dim=1)+eps) +
                 2*(1-weight) * torch.sqrt(torch.sum((Y_obs_bj - Y_obs) ** 2, dim=1)
                                           + eps)) ** 2
    else:
        inner = (2 * weight * torch.sqrt(
            torch.sum(M_obs * (X_obs - Y_obs) ** 2, dim=1) + eps) +
                 2 * (1 - weight) * torch.sqrt(
                    torch.sum(M_obs * (Y_obs_bj - Y_obs) ** 2, dim=1)
                    + eps)) ** 2
    outer = torch.sum(inner / n_obs_ot)
    return outer / batch_size


def compute_loss_2(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
                   weight=0.5, M_obs=None):
    """
    similar to compute_loss, but using X_obs also in second part of loss
    instead of Y_obs
    """
    if M_obs is None:
        inner = (weight * torch.sqrt(torch.sum((X_obs - Y_obs) ** 2, dim=1) + eps) +
                 (1 - weight) * torch.sqrt(torch.sum((Y_obs_bj - X_obs) ** 2, dim=1)
                                           + eps)) ** 2
    else:
        inner = (weight * torch.sqrt(
            torch.sum(M_obs * (X_obs - Y_obs) ** 2, dim=1) + eps) +
                 (1 - weight) * torch.sqrt(
                    torch.sum(M_obs * (Y_obs_bj - X_obs) ** 2, dim=1)
                    + eps)) ** 2
    outer = torch.sum(inner / n_obs_ot)
    return outer / batch_size


LOSS_FUN_DICT = {
    'standard': compute_loss,
    'easy': compute_loss_2,
}

nonlinears = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU
}


def get_ffnn(input_size, output_size, nn_desc, dropout_rate, bias):
    """
    function to get a feed-forward neural network with the given description
    :param input_size: int, input dimension
    :param output_size: int, output dimension
    :param nn_desc: list of lists or None, each inner list defines one hidden
            layer and has 2 elements: 1. int, the hidden dim, 2. str, the
            activation function that should be applied (see dict nonlinears for
            possible options)
    :param dropout_rate: float,
    :param bias: bool, whether a bias is used in the layers
    :return: torch.nn.Sequential, the NN function
    """
    if nn_desc is None:
        layers = [torch.nn.Linear(input_size, output_size, bias=bias)]
    else:
        layers = [torch.nn.Linear(input_size, nn_desc[0][0], bias=bias)]
        if len(nn_desc) > 1:
            for i in range(len(nn_desc)-1):
                layers.append(nonlinears[nn_desc[i][1]]())
                layers.append(torch.nn.Dropout(p=dropout_rate))
                layers.append(torch.nn.Linear(nn_desc[i][0], nn_desc[i+1][0],
                                              bias=bias))
        layers.append(nonlinears[nn_desc[-1][1]]())
        layers.append(torch.nn.Dropout(p=dropout_rate))
        layers.append(torch.nn.Linear(nn_desc[-1][0], output_size, bias=bias))
    return torch.nn.Sequential(*layers)


# =====================================================================================================================
class ODEFunc(torch.nn.Module):
    """
    implementing continuous update between observatios, f_{\theta} in paper
    """
    def __init__(self, input_size, hidden_size, ode_nn, dropout_rate=0.0,
                 bias=True):
        super().__init__()

        # create feed-forward NN, f(H,X,tau,t-tau)
        self.f = get_ffnn(
            input_size=input_size+hidden_size+2, output_size=hidden_size,
            nn_desc=ode_nn, dropout_rate=dropout_rate, bias=bias
        )

    def forward(self, x, h, tau, tdiff):
        # dimension should be (batch, input_size) for x, (batch, hidden) for h, 
        #    (batch, 1) for times
        input_f = torch.cat([torch.tanh(x), torch.tanh(h), tau, tdiff], dim=1)
        df = self.f(input_f)
        return df


class GRUCell(torch.nn.Module):
    """
    Implements discrete update based on the received observations, \rho_{\theta}
    in paper
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.gru_d = torch.nn.GRUCell(input_size, hidden_size, bias=bias)

        self.input_size = input_size

    def forward(self, h, X_obs, i_obs):
        temp = h.clone()
        temp[i_obs] = self.gru_d(torch.tanh(X_obs), torch.tanh(h[i_obs]))
        h = temp
        return h


class FFNN(torch.nn.Module):
    """
    Implements feed-forward neural networks with tanh applied to inputs and the
    option to use a residual NN version (then the output size needs to be a
    multiple of the input size or vice versa)
    """
    def __init__(self, input_size, output_size, nn_desc, dropout_rate=0.0,
                 bias=True, residual=False, masked=False):
        super().__init__()

        # create feed-forward NN
        in_size = input_size
        if masked:
            in_size = 2*input_size
        self.masked = masked
        self.ffnn = get_ffnn(
            input_size=in_size, output_size=output_size,
            nn_desc=nn_desc, dropout_rate=dropout_rate, bias=bias
        )

        if residual:
            print('use residual network: input_size={}, output_size={}'.format(
                input_size, output_size))
            if input_size <= output_size:
                if output_size % input_size == 0:
                    self.case = 1
                    self.mult = int(output_size / input_size)
                else:
                    raise ValueError('for residual: output_size needs to be '
                                     'multiple of input_size')

            if input_size > output_size:
                if input_size % output_size == 0:
                    self.case = 2
                    self.mult = int(input_size / output_size)
                else:
                    raise ValueError('for residual: input_size needs to be '
                                     'multiple of output_size')
        else:
            self.case = 0
            
    def forward(self, nn_input, mask=None):
        if self.masked:
            assert mask is not None
            out = self.ffnn(torch.cat((torch.tanh(nn_input), mask), 1))
        else:
            out = self.ffnn(torch.tanh(nn_input))

        if self.case == 0:
            return out
        elif self.case == 1:
            identity = nn_input.repeat(1, self.mult)
            return identity + out
        elif self.case == 2:
            identity = torch.mean(torch.stack(nn_input.chunk(self.mult, dim=1)),
                                  dim=0)
            return identity + out



class NJODE(torch.nn.Module):
    """
    NJ-ODE model
    """
    def __init__(
            self, input_size, hidden_size, output_size,
            ode_nn, readout_nn, enc_nn, use_rnn,
            bias=True, dropout_rate=0, solver="euler",
            weight=0.5, weight_decay=1.,
            **options
    ):
        """
        init the model
        :param input_size: int
        :param hidden_size: int, size of latent variable process
        :param output_size: int
        :param ode_nn: list of list, defining the NN f, see get_ffnn
        :param readout_nn: list of list, defining the NN g, see get_ffnn
        :param enc_nn: list of list, defining the NN e, see get_ffnn
        :param use_rnn: bool, whether to use the RNN for 'jumps'
        :param bias: bool, whether to use a bias for the NNs
        :param dropout_rate: float
        :param solver: str, specifying the ODE solver, suppoorted: {'euler'}
        :param weight: float in [0.5, 1], the initial weight used in the loss
        :param weight_decay: float in [0,1], the decay applied to the weight of
                the loss function after each epoch, decaying towards 0.5
                    1: no decay, weight stays the same
                    0: immediate decay to 0.5 after 1st epoch
                    (0,1): exponential decay towards 0.5
        :param options: kwargs, used: kword "options" with arg a dict passed
                from train.train (kwords: 'which_loss', 'residual_enc_dec',
                'masked' are used)
        """
        super().__init__()

        self.epoch = 1
        self.weight = weight
        self.weight_decay = weight_decay
        self.use_rnn = use_rnn
        
        # get options from the optians of train input
        options1 = options['options']
        if 'which_loss' in options1:
            self.which_loss = options1['which_loss']
        else:
            self.which_loss = 'standard'
        assert self.which_loss in LOSS_FUN_DICT
        print('using loss: {}'.format(self.which_loss))
        
        if 'residual_enc_dec' in options1:
            self.residual_enc_dec = options1['residual_enc_dec']
        else:
            self.residual_enc_dec = True

        self.masked = False
        if 'masked' in options1:
            self.masked = options1['masked']
        
        self.ode_f = ODEFunc(input_size, hidden_size, ode_nn, dropout_rate, bias)
        self.encoder_map = FFNN(
            input_size, hidden_size, enc_nn, dropout_rate, bias,
            masked=self.masked,
            residual=self.residual_enc_dec)
        self.readout_map = FFNN(
            hidden_size, output_size, readout_nn, dropout_rate, bias, 
            residual=self.residual_enc_dec)
        if self.use_rnn:  # TODO: implement that also this can be used with mask
            self.obs_c = GRUCell(input_size, hidden_size, bias=bias)

        # assert (solver in ["euler"], "Solver must be 'euler'.")

        self.solver = solver
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.apply(init_weights)

    def weight_decay_step(self):
        inc = (self.weight - 0.5)
        self.weight = 0.5 + inc * self.weight_decay
        return self.weight

    def ode_step(self, h, delta_t, current_time, last_X, tau):
        """Executes a single ODE step"""
        if self.solver == "euler":
            h = h + delta_t * self.ode_f(last_X, h, tau, current_time - tau)
        else:
            raise ValueError("Unknown solver '{}'.".format(self.solver))

        current_time += delta_t
        return h, current_time

    def forward(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                n_obs_ot, return_path=False, get_loss=True, until_T=False,
                M=None):
        """
        the forward run of this module class, used when calling the module
        instance without a method
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: torch.tensor, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: torch.tensor, the starting point of X
        :param n_obs_ot: torch.tensor, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or torch.tensor, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :return: torch.tensor (hidden state at final time), torch.tensor (loss),
                    if wanted the paths of t (np.array) and h, y (torch.tensors)
        """

        if self.masked:
            h = self.encoder_map(start_X, mask=torch.zeros_like(start_X))
        else:
            h = self.encoder_map(start_X)

        last_X = start_X
        batch_size = start_X.size()[0]
        tau = torch.tensor([[0.0]]).repeat(batch_size, 1)
        current_time = 0.0

        loss = 0

        if return_path:
            path_t = [0]
            path_h = [h]
            path_y = [self.readout_map(h)]

        assert len(times) + 1 == len(time_ptr)

        for i, obs_time in enumerate(times):
            # Propagation of the ODE until next observation
            while current_time < (obs_time - 1e-10 * delta_t):  # 0.0001 delta_t used for numerical consistency.
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(h, delta_t_, current_time, 
                                                    last_X=last_X, tau=tau)

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    path_y.append(self.readout_map(h))

            # Reached an observation - only update those elements of the batch, 
            #    for which an observation is made
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            if self.masked:
                M_obs = M[start:end]
            else:
                M_obs = None

            # Using RNNCell to update h. Also updating loss, tau and last_X
            Y_bj = self.readout_map(h)
            if self.use_rnn:
                h = self.obs_c(h, X_obs, i_obs)
            else:
                temp = h.clone()
                if self.masked:
                    X_obs_impute = X_obs * M_obs + \
                                   (torch.ones_like(M_obs) - M_obs)*Y_bj[i_obs]
                    temp[i_obs] = self.encoder_map(X_obs_impute, M_obs)
                else:
                    temp[i_obs] = self.encoder_map(X_obs)
                h = temp
            Y = self.readout_map(h)
            
            if get_loss:
                loss = loss + LOSS_FUN_DICT[self.which_loss](
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=self.weight, M_obs=M_obs)

            # make update of last_X and tau, that is not inplace 
            #    (otherwise problems in autograd)
            temp_X = last_X.clone()
            temp_tau = tau.clone()
            if self.masked:
                temp_X[i_obs] = Y[i_obs]
            else:
                temp_X[i_obs] = X_obs
            temp_tau[i_obs] = obs_time.astype(np.float64)
            last_X = temp_X
            tau = temp_tau

            if return_path:
                path_t.append(obs_time)
                path_h.append(h)
                path_y.append(Y)

        # after every observation has been processed, propagating until T
        if until_T:
            while current_time < T - 1e-10 * delta_t:
                if current_time < T - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = T - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(h, delta_t_, current_time, 
                                                    last_X=last_X, tau=tau)

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    path_y.append(self.readout_map(h))

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return h, loss, np.array(path_t), torch.stack(path_h), \
                   torch.stack(path_y)
        else:
            return h, loss


    def evaluate(self, times, time_ptr, X, obs_idx, delta_t, T, start_X, 
                 n_obs_ot, stockmodel, cond_exp_fun_kwargs=None,
                 diff_fun=lambda x,y: np.mean((x-y)**2),
                 return_paths=False, M=None):
        """
        evaluate the model at its current training state against the true
        conditional expectation
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param n_obs_ot: see forward
        :param stockmodel: stock_model.StockModel instance, used to compute true
                cond. exp.
        :param cond_exp_fun_kwargs: dict, the kwargs for the cond. exp. function
                currently not used
        :param diff_fun: function, to compute difference between optimal and
                predicted cond. exp
        :param return_paths: bool, whether to return also the paths
        :param M: see forward
        :return: eval-loss, if wanted paths t, y for true and pred
        """
        self.eval()
        _, _, path_t, path_h, path_y = self.forward(
            times, time_ptr,X,obs_idx, delta_t, T, start_X, None, 
            return_path=True, get_loss=False, until_T=True, M=M)

        loss, true_path_t, true_path_y = stockmodel.compute_cond_exp(
            times, time_ptr, X.detach().numpy(), obs_idx.detach().numpy(),
            delta_t, T, start_X.detach().numpy(), n_obs_ot.detach().numpy(),
            return_path=True, get_loss=False)
        eval_loss = diff_fun(path_y.detach().numpy(), true_path_y)
        if return_paths:
            return eval_loss, path_t, true_path_t, path_y, true_path_y
        else:
            return eval_loss

    def get_pred(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                 M=None):
        """
        get predicted path
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param M: see forward
        :return: dict, with prediction y and times t
        """
        self.eval()
        _, _, path_t, path_h, path_y = self.forward(times, time_ptr, X, obs_idx,
                                                    delta_t, T, start_X, None,
                                                    return_path=True,
                                                    get_loss=False,
                                                    until_T=True, M=M)
        return {'pred': path_y, 'pred_t': path_t}
    



