"""
author: Florian Krach & Calypso Herrera

stock models to create the datasets
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
                 maturity, **kwargs):
        self.drift = drift
        self.volatility = volatility
        self.S0 = S0
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.dimensions = np.size(S0)

    def generate_paths(self):
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
                         weight=0.5, **kwargs):
        """
        compute conditional expectation similar to computing the prediction in
        the model.CODERNN.forward
        :param times: see model.CODERNN.forward
        :param time_ptr: see model.CODERNN.forward
        :param X: see model.CODERNN.forward, as np.array
        :param obs_idx: see model.CODERNN.forward, as np.array
        :param delta_t: see model.CODERNN.forward, as np.array
        :param T: see model.CODERNN.forward
        :param start_X: see model.CODERNN.forward, as np.array
        :param n_obs_ot: see model.CODERNN.forward, as np.array
        :param return_path: see model.CODERNN.forward
        :param get_loss: see model.CODERNN.forward
        :param weight: see model.CODERNN.forward
        :param kwargs: unused, to allow for additional unused inputs
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        y = start_X
        batch_size = start_X.shape[0]
        current_time = 0.0

        loss = 0

        if return_path:
            path_t = [0.]
            path_y = [y]

        for i, obs_time in enumerate(times):
            # Propagation of the ODE until next observation
            while current_time < (
                    obs_time - 1e-10 * delta_t):  # 1e-10*delta_t used for numerical consistency.
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_)
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
                 nb_steps, S0, maturity, **kwargs):
        super(Heston, self).__init__(
            drift=drift, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps,
            S0=S0, maturity=maturity)
        self.mean = mean
        self.speed = speed
        self.correlation = correlation

    def next_cond_exp(self, y, delta_t):
        return y * np.exp(self.drift * delta_t)

    def generate_paths(self):
        # Diffusion of the spot: dS = mu*S*dt + sqrt(v)*S*dW
        spot_drift = lambda x, t: self.drift * x
        spot_diffusion = lambda x, v, t: np.sqrt(v) * x

        # Diffusion of the variance: dv = -k(v-vinf)*dt + sqrt(v)*v*dW
        var_drift = lambda v, t: - self.speed * (v - self.mean)
        var_diffusion = lambda v, t: self.volatility * np.sqrt(v)

        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        var_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))

        dt = self.maturity / self.nb_steps
        for i in range(self.nb_paths):
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
                        + np.dot(
                    var_diffusion(var_paths[i, :, k - 1], (k) * dt), dZ))

                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + spot_drift(spot_paths[i, :, k - 1], (k) * dt) * dt
                        + np.dot(spot_diffusion(spot_paths[i, :, k - 1],
                                                var_paths[i, :, k],
                                                (k) * dt), dW))

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


class BlackScholes(StockModel):
    """
    standard Black-Scholes model, see:
    https://en.wikipedia.org/wiki/Black–Scholes_model
    https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    """
    def __init__(self, drift, volatility, nb_paths, nb_steps, S0,
                 maturity, **kwargs):
        super(BlackScholes, self).__init__(
            drift=drift, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=S0, maturity=maturity)

    def next_cond_exp(self, y, delta_t):
        return y * np.exp(self.drift * delta_t)

    def generate_paths(self):
        drift = lambda x, t: self.drift * x
        diffusion = lambda x, t: self.volatility * x
        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps
        for i in range(self.nb_paths):
            spot_paths[i, :, 0] = self.S0
            for k in range(1, self.nb_steps + 1):
                random_numbers = np.random.normal(0, 1, self.dimensions)
                dW = random_numbers * np.sqrt(dt)
                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + drift(spot_paths[i, :, k - 1], (k) * dt) * dt
                        + np.dot(diffusion(spot_paths[i, :, k - 1], (k) * dt),
                                 dW))
        return spot_paths, dt


class OrnsteinUhlenbeck(StockModel):
    """
    Ornstein-Uhlenbeeck stock model, see:
    https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
    """
    def __init__(self, volatility, nb_paths, nb_steps, S0,
                 mean, speed, maturity, **kwargs):
        super(OrnsteinUhlenbeck, self).__init__(
            volatility=volatility, nb_paths=nb_paths, drift=None,
            nb_steps=nb_steps, S0=S0, maturity=maturity)
        self.mean = mean
        self.speed = speed

    def next_cond_exp(self, y, delta_t):
        exp_delta = np.exp(-self.speed * delta_t)
        return y * exp_delta + self.mean * (1 - exp_delta)

    def generate_paths(self):
        # Diffusion of the variance: dv = -k(v-vinf)*dt + vol*dW
        drift = lambda x, t: - self.speed * (x - self.mean)
        diffusion = lambda x, t: self.volatility

        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.maturity / self.nb_steps
        for i in range(self.nb_paths):
            spot_paths[i, :, 0] = self.S0
            for k in range(1, self.nb_steps + 1):
                random_numbers = np.random.normal(0, 1, self.dimensions)
                dW = random_numbers * np.sqrt(dt)
                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + drift(spot_paths[i, :, k - 1], (k) * dt) * dt
                        + np.dot(diffusion(spot_paths[i, :, k - 1], (k) * dt),
                                 dW))
        return spot_paths, dt


# ==============================================================================
# this is needed for computiing the loss with the true conditional expectation
def compute_loss(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
                 weight=0.5):
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
    "OrnsteinUhlenbeck": OrnsteinUhlenbeck
}




