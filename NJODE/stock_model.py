"""
author: Florian Krach & Calypso Herrera

code to generate synthetic data from stock-model SDEs
"""

# ==============================================================================
from math import sqrt, exp
import numpy as np
import matplotlib.pyplot as plt
import copy, os


# ==============================================================================
class StockModel:
    """
    mother class for all stock models defining the variables and methods shared
    amongst all of them, some need to be defined individually
    """
    def __init__(self, drift, volatility, S0, nb_paths, nb_steps,
                 maturity, sine_coeff, **kwargs):
        self.drift = drift
        self.volatility = volatility
        self.S0 = S0
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.dimensions = np.size(S0)
        if sine_coeff is None:
            self.periodic_coeff = lambda t: 1
        else:
            self.periodic_coeff = lambda t: (1 + np.sin(sine_coeff * t))


    def generate_paths(self, **options):
        """
        generate random paths according to the model hyperparams
        :return: stock paths as np.array, dim: [nb_paths, data_dim, nb_steps]
        """
        raise ValueError("not implemented yet")

    def next_cond_exp(self, *args, **kwargs):
        """
        compute the next point of the conditional expectation starting from
        given point for given time_delta
        :return: cond. exp. at next time_point (= current_time + time_delta)
        """
        raise ValueError("not implemented yet")

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5,
                         start_time=None,
                         **kwargs):
        """
        compute conditional expectation similar to computing the prediction in
        the model.NJODE.forward
        :param times: see model.NJODE.forward
        :param time_ptr: see model.NJODE.forward
        :param X: see model.NJODE.forward, as np.array
        :param obs_idx: see model.NJODE.forward, as np.array
        :param delta_t: see model.NJODE.forward, as np.array
        :param T: see model.NJODE.forward
        :param start_X: see model.NJODE.forward, as np.array
        :param n_obs_ot: see model.NJODE.forward, as np.array
        :param return_path: see model.NJODE.forward
        :param get_loss: see model.NJODE.forward
        :param weight: see model.NJODE.forward
        :param start_time: None or float, if float, this is first time point
        :param kwargs: unused, to allow for additional unused inputs
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        y = start_X
        batch_size = start_X.shape[0]
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            if obs_time > T + 1e-10:
                break
            if obs_time <= current_time:
                continue
            # Propagation of the ODE until next observation
            while current_time < (
                    obs_time - 1e-10 * delta_t):  # 1e-10*delta_t used for numerical consistency.
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                current_time = current_time + delta_t_

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - only update those elements of the batch, 
            #    for which an observation is made
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]

            # Using RNNCell to update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            temp[i_obs] = X_obs
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(X_obs=X_obs, Y_obs=Y[i_obs],
                                           Y_obs_bj=Y_bj[i_obs],
                                           n_obs_ot=n_obs_ot[i_obs],
                                           batch_size=batch_size, weight=weight)

            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss

    def get_optimal_loss(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, weight=0.5):
        loss = self.compute_cond_exp(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            return_path=False, get_loss=True, weight=weight)
        return loss


class Heston(StockModel):
    """
    the Heston model, see: https://en.wikipedia.org/wiki/Heston_model
    a basic stochastic volatility stock price model
    """
    def __init__(self, drift, volatility, mean, speed, correlation, nb_paths,
                 nb_steps, S0, maturity, sine_coeff=None, **kwargs):
        super(Heston, self).__init__(
            drift=drift, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps,
            S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.mean = mean
        self.speed = speed
        self.correlation = correlation

    def next_cond_exp(self, y, delta_t, current_t):
        return y * np.exp(self.drift*self.periodic_coeff(current_t)*delta_t)

    def generate_paths(self, start_X=None):
        # Diffusion of the spot: dS = mu*S*dt + sqrt(v)*S*dW
        spot_drift = lambda x, t: self.drift*self.periodic_coeff(t)*x
        spot_diffusion = lambda x, v, t: np.sqrt(v) * x

        # Diffusion of the variance: dv = -k(v-vinf)*dt + sqrt(v)*v*dW
        var_drift = lambda v, t: - self.speed * (v - self.mean)
        var_diffusion = lambda v, t: self.volatility * np.sqrt(v)

        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        var_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))

        dt = self.maturity / self.nb_steps
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        for i in range(self.nb_paths):
            if start_X is None:
                spot_paths[i, :, 0] = self.S0
            var_paths[i, :, 0] = self.mean
            for k in range(1, self.nb_steps + 1):
                normal_numbers_1 = np.random.normal(0, 1, self.dimensions)
                normal_numbers_2 = np.random.normal(0, 1, self.dimensions)
                dW = normal_numbers_1 * np.sqrt(dt)
                dZ = (self.correlation * normal_numbers_1 + np.sqrt(
                    1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(dt)

                var_paths[i, :, k] = (
                        var_paths[i, :, k - 1]
                        + var_drift(var_paths[i, :, k - 1], (k) * dt) * dt
                        + var_diffusion(var_paths[i, :, k - 1], (k) * dt) * dZ)

                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + spot_drift(spot_paths[i, :, k - 1], (k-1) * dt) * dt
                        + spot_diffusion(spot_paths[i, :, k - 1],
                                                var_paths[i, :, k],
                                                (k) * dt) * dW)
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt

    def draw_path_heston(self, filename):
        nb_paths = self.nb_paths
        self.nb_paths = 1
        paths, dt = self.generate_paths()
        self.nb_paths = nb_paths
        spot_paths, var_paths = paths
        one_spot_path = spot_paths[0, 0, :]
        one_var_path = var_paths[0, 0, :]
        dates = np.array([i for i in range(len(one_spot_path))])
        dt = self.maturity / self.nb_steps
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('time')
        ax1.set_ylabel('Stock', color=color)
        ax1.plot(dates, one_spot_path, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.set_ylabel('Variance', color=color)
        ax2.plot(dates, one_var_path, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.savefig(filename)
        plt.close()



class HestonWOFeller(StockModel):
    """
    the Heston model, see: https://en.wikipedia.org/wiki/Heston_model
    a basic stochastic volatility stock price model, that can be used
    even if Feller condition is not satisfied
    Feller condition: 2*speed*mean > volatility**2
    """
    def __init__(self, drift, volatility, mean, speed, correlation, nb_paths,
                 nb_steps, S0, maturity, scheme='euler', return_vol=False,
                 v0=None, sine_coeff=None, **kwargs):
        super(HestonWOFeller, self).__init__(
            drift=drift, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps,
            S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.mean = mean
        self.speed = speed
        self.correlation = correlation

        self.scheme = scheme
        self.retur_vol = return_vol
        if v0 is None:
            self.v0 = self.mean
        else:
            self.v0 = v0

    def next_cond_exp(self, y, delta_t, current_t):
        if self.retur_vol:
            s, v = np.split(y, indices_or_sections=2, axis=1)
            s = s * np.exp(self.drift*self.periodic_coeff(current_t)*delta_t)
            exp_delta = np.exp(-self.speed * delta_t)
            v = v * exp_delta + self.mean * (1 - exp_delta)
            y = np.concatenate([s, v], axis=1)
            return y
        else:
            return y*np.exp(self.drift*self.periodic_coeff(current_t)*delta_t)

    def generate_paths(self, start_X=None):
        if self.scheme == 'euler':
            # Diffusion of the spot: dS = mu*S*dt + sqrt(v)*S*dW
            log_spot_drift = lambda v, t: \
                (self.drift*self.periodic_coeff(t) - 0.5 * np.maximum(v, 0))
            log_spot_diffusion = lambda v: np.sqrt(np.maximum(v, 0))

            # Diffusion of the variance: dv = -k(v-vinf)*dt + sqrt(v)*v*dW
            var_drift = lambda v: - self.speed * (np.maximum(v, 0) - self.mean)
            var_diffusion = lambda v: self.volatility * np.sqrt(np.maximum(v, 0))

            spot_paths = np.empty(
                (self.nb_paths, self.dimensions, self.nb_steps + 1))
            var_paths = np.empty(
                (self.nb_paths, self.dimensions, self.nb_steps + 1))

            dt = self.maturity / self.nb_steps
            if start_X is not None:
                spot_paths[:, :, 0] = start_X
            for i in range(self.nb_paths):
                if start_X is None:
                    spot_paths[i, :, 0] = self.S0
                var_paths[i, :, 0] = self.v0
                for k in range(1, self.nb_steps + 1):
                    normal_numbers_1 = np.random.normal(0, 1, self.dimensions)
                    normal_numbers_2 = np.random.normal(0, 1, self.dimensions)
                    dW = normal_numbers_1 * np.sqrt(dt)
                    dZ = (self.correlation * normal_numbers_1 + np.sqrt(
                        1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(dt)

                    spot_paths[i, :, k] = np.exp(
                            np.log(spot_paths[i, :, k - 1])
                            + log_spot_drift(
                                var_paths[i, :, k - 1], (k-1)*dt) * dt
                            + log_spot_diffusion(var_paths[i, :, k - 1]) * dW
                    )
                    var_paths[i, :, k] = (
                            var_paths[i, :, k - 1]
                            + var_drift(var_paths[i, :, k - 1]) * dt
                            + var_diffusion(var_paths[i, :, k - 1]) * dZ
                    )
            if self.retur_vol:
                spot_paths = np.concatenate([spot_paths, var_paths], axis=1)
            # stock_path dimension: [nb_paths, dimension, time_steps]
            return spot_paths, dt

        else:
            raise ValueError('unknown sampling scheme')



class BlackScholes(StockModel):
    """
    standard Black-Scholes model, see:
    https://en.wikipedia.org/wiki/Black–Scholes_model
    https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    """
    def __init__(self, drift, volatility, nb_paths, nb_steps, S0,
                 maturity, sine_coeff=None, **kwargs):
        super(BlackScholes, self).__init__(
            drift=drift, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )

    def next_cond_exp(self, y, delta_t, current_t):
        return y * np.exp(self.drift*self.periodic_coeff(current_t)*delta_t)

    def generate_paths(self, start_X=None):
        drift = lambda x, t: self.drift*self.periodic_coeff(t)*x
        diffusion = lambda x, t: self.volatility * x
        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        for i in range(self.nb_paths):
            if start_X is None:
                spot_paths[i, :, 0] = self.S0
            for k in range(1, self.nb_steps + 1):
                random_numbers = np.random.normal(0, 1, self.dimensions)
                dW = random_numbers * np.sqrt(dt)
                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + drift(spot_paths[i, :, k - 1], (k-1) * dt) * dt
                        + diffusion(spot_paths[i, :, k - 1], (k) * dt) * dW)
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class OrnsteinUhlenbeck(StockModel):
    """
    Ornstein-Uhlenbeeck stock model, see:
    https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
    """
    def __init__(self, volatility, nb_paths, nb_steps, S0,
                 mean, speed, maturity, sine_coeff=None, **kwargs):
        super(OrnsteinUhlenbeck, self).__init__(
            volatility=volatility, nb_paths=nb_paths, drift=None,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.mean = mean
        self.speed = speed

    def next_cond_exp(self, y, delta_t, current_t):
        exp_delta = np.exp(-self.speed*self.periodic_coeff(current_t)*delta_t)
        return y * exp_delta + self.mean * (1 - exp_delta)

    def generate_paths(self, start_X=None):
        # Diffusion of the variance: dv = -k(v-vinf)*dt + vol*dW
        drift = lambda x, t: - self.speed*self.periodic_coeff(t)*(x - self.mean)
        diffusion = lambda x, t: self.volatility

        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        for i in range(self.nb_paths):
            if start_X is None:
                spot_paths[i, :, 0] = self.S0
            for k in range(1, self.nb_steps + 1):
                random_numbers = np.random.normal(0, 1, self.dimensions)
                dW = random_numbers * np.sqrt(dt)
                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + drift(spot_paths[i, :, k - 1], (k-1) * dt) * dt
                        + diffusion(spot_paths[i, :, k - 1], (k) * dt) * dW)
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class Combined(StockModel):
    def __init__(self, stock_model_names, hyperparam_dicts, **kwargs):
        self.stock_model_names = stock_model_names
        self.hyperparam_dicts = hyperparam_dicts

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, **kwargs):
        # get first stockmodel
        stockmodel = STOCK_MODELS[self.stock_model_names[0]](
            **self.hyperparam_dicts[0])
        T = self.hyperparam_dicts[0]['maturity']
        loss, path_t, path_y = stockmodel.compute_cond_exp(
            times, time_ptr, X, obs_idx, delta_t,
            T, start_X,
            n_obs_ot, return_path=True, get_loss=get_loss,
            weight=weight,
        )
        for i in range(1, len(self.stock_model_names)):
            start_X = path_y[-1, :, :]
            start_time = path_t[-1]
            T += self.hyperparam_dicts[i]['maturity']
            stockmodel = STOCK_MODELS[self.stock_model_names[i]](
                **self.hyperparam_dicts[i])
            _loss, _path_t, _path_y = stockmodel.compute_cond_exp(
                times, time_ptr, X, obs_idx, delta_t,
                T, start_X,
                n_obs_ot, return_path=True, get_loss=get_loss,
                weight=weight, start_time=start_time
            )
            loss += _loss
            path_t = np.concatenate([path_t, _path_t])
            path_y = np.concatenate([path_y, _path_y], axis=0)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss

    def get_optimal_loss(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, weight=0.5):
        loss = self.compute_cond_exp(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            return_path=False, get_loss=True, weight=weight)
        return loss


# ==============================================================================
# this is needed for computing the loss with the true conditional expectation
def compute_loss(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
                 weight=0.5):
    """
    compute the loss of the true conditional expectation, as in
    model.compute_loss
    """
    inner = (2 * weight * np.sqrt(np.sum((X_obs - Y_obs) ** 2, axis=1) + eps) +
             2 * (1 - weight) * np.sqrt(np.sum((Y_obs_bj - Y_obs) ** 2, axis=1)
                                        + eps)) ** 2
    outer = np.sum(inner / n_obs_ot)
    return outer / batch_size


# ==============================================================================
# dict for the supported stock models to get them from their name
STOCK_MODELS = {
    "BlackScholes": BlackScholes,
    "Heston": Heston,
    "OrnsteinUhlenbeck": OrnsteinUhlenbeck,
    "HestonWOFeller": HestonWOFeller,
    "combined": Combined,
    "sine_BlackScholes": BlackScholes,
    "sine_Heston": Heston,
    "sine_OrnsteinUhlenbeck": OrnsteinUhlenbeck,
}
# ==============================================================================


hyperparam_test_stock_models = {
    'drift': 0.2, 'volatility': 0.3, 'mean': 0.5,
    'speed': 0.5, 'correlation': 0.5, 'nb_paths': 10, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1}


def draw_stock_model(stock_model_name):
    hyperparam_test_stock_models['model_name'] = stock_model_name
    stockmodel = STOCK_MODELS[stock_model_name](**hyperparam_test_stock_models)
    stock_paths, dt = stockmodel.generate_paths()
    filename = '{}.pdf'.format(stock_model_name)

    # draw a path
    one_path = stock_paths[0, 0, :]
    dates = np.array([i for i in range(len(one_path))])
    cond_exp = np.zeros(len(one_path))
    cond_exp[0] = hyperparam_test_stock_models['S0']
    cond_exp_const = hyperparam_test_stock_models['S0']
    for i in range(1, len(one_path)):
        if i % 3 == 0:
            cond_exp[i] = one_path[i]
        else:
            cond_exp[i] = cond_exp[i - 1] * exp(
                hyperparam_test_stock_models['drift'] * dt)

    plt.plot(dates, one_path, label='stock path')
    plt.plot(dates, cond_exp, label='conditional expectation')
    plt.legend()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    draw_stock_model("BlackScholes")
    heston = STOCK_MODELS["Heston"](**hyperparam_test_stock_models)
    heston.draw_path_heston("heston.pdf")
