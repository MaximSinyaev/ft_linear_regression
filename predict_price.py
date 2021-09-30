# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    predict_price.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aolen <aolen@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2021/09/30 23:35:16 by aolen             #+#    #+#              #
#    Updated: 2021/10/01 00:02:22 by aolen            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from argparse import ArgumentParser

from model import RegressionModel

def construct_parser():
    parser = ArgumentParser(prog='train_model', description="Util for training linear regression")
    parser.add_argument("model-path", help='Path for saved model json or pickle', type=str)

    return parser

if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()

    model = RegressionModel.load(vars(args)['model-path'])
    x = float(input("For what milage should we predict price? "))
    print(model.predict(x))