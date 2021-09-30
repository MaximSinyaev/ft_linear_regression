# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aolen <aolen@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2021/09/30 22:23:47 by aolen             #+#    #+#              #
#    Updated: 2021/10/01 00:13:36 by aolen            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import json
import pickle
from collections import defaultdict

import numpy as np

from metrics import mse, mae

class RegressionModel:
    """
    RegressionModel common class for model training and predicting
    """
    file_config_types = ['json', 'pickle']

    def __init__(
            self,
            theta0=None,
            theta1=None,
            lr=7e-3,
            tol=1e-5,
            debug=False
    ):
        self.theta0 = theta0
        self.theta1 = theta1
        self.tol = tol
        self.lr = lr
        self.debug = debug

    def _add_loss(self, loss_dict, mse, mae):
        loss_dict['MSE'].append(mse)
        loss_dict['MAE'].append(mae)

    def fit(self, X, y, epochs=5000):
        """
        Func for model train
        """
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)

        self.theta0 = 0
        self.theta1 = float(np.random.rand(1))

        loss_dict = defaultdict(list)

        prev_loss = float('inf')
        y_pred = self.predict(X)
        loss = mse(y, y_pred)
        mae_loss = mae(y, y_pred)
        self._add_loss(loss_dict, loss, mae_loss)

        if self.debug:
            print(f'Init MSE loss: {loss:.3f}, MAE loss: {mae_loss:.3f}')

        epoch_num = 0
        while abs(loss - prev_loss) > self.tol and epoch_num < epochs:
            epoch_num += 1
            prev_loss = loss
            self.theta1 -= self.lr * np.mean(X * (y_pred - y))
            self.theta0 -= self.lr * np.mean(y_pred - y)
            y_pred = self.predict(X)
            loss = mse(y, y_pred)
            mae_loss = mae(y, y_pred)
            self._add_loss(loss_dict, loss, mae_loss)
            if self.debug:
                print(f'For {epoch_num} epoch MSE loss: {loss:.3f}, MAE loss: {mae_loss:.3f}')

        print(self.theta1)
        return loss_dict

    def predict(self, X, y=None):
        assert self.theta1 is not None  # "Model is not trained"

        y_pred = self.theta0 + self.theta1 * X

        return y_pred

    @classmethod
    def load(cls, path, config_type='json'):
        if config_type not in cls.file_config_types:
            raise TypeError('Wrong file type provided')
        if config_type == 'json':
            with open(path, 'r') as json_file:
                params = json.load(json_file)
                model = cls(theta0=params['theta0'], theta1=params['theta1'])
        elif config_type == 'pickle':
            with open(path, 'rb') as pickle_file:
                model = pickle.load(pickle_file)
        return model

    def save(self, path, config_type='json'):
        if config_type not in self.file_config_types:
            raise TypeError('Wrong file type provided')
        if config_type == 'json':
            with open(path, 'w') as json_file:
                params = {'theta0': self.theta0, 'theta1': self.theta1}
                json.dump(params, json_file)
        elif config_type == 'pickle':
            with open(path, 'wb') as pickle_file:
                pickle.dump(self, pickle_file)