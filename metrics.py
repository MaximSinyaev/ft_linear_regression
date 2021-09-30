# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    metrics.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aolen <aolen@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2021/09/30 22:44:45 by aolen             #+#    #+#              #
#    Updated: 2021/09/30 22:44:51 by aolen            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))