# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_model.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aolen <aolen@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2021/09/30 22:45:14 by aolen             #+#    #+#              #
#    Updated: 2021/10/01 00:08:47 by aolen            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from argparse import ArgumentParser
from collections import defaultdict
import csv

import matplotlib.pyplot as plt

from model import RegressionModel
from utils import scale_data


def read_data(data_path):
    dataset = defaultdict(list)

    with open(data_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader):
            if i == 0:
                cols = row
                continue
            for i, val in enumerate(row):
                dataset[cols[i]].append(float(val))
    
    return dataset

def construct_parser():
    parser = ArgumentParser(prog='train_model', description="Util for training linear regression")
    parser.add_argument("dataset-path", help='Path for csv dataset with 2 data columns', type=str)
    parser.add_argument("--verbose", help='Show learning progress', action='store_true')
    parser.add_argument("--display-metric", help='If specified generate plot with loss function', action='store_true')
    parser.add_argument("--lr", help='Learning rate value', type=float, default=0.007)
    parser.add_argument("--max-epochs", help='Maximum number of epoches', type=int, default=5000)

    return parser

if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    model = RegressionModel(
        lr=args.lr,
        debug=args.verbose
    )

    dataset = read_data(vars(args)['dataset-path'])
    X = scale_data(dataset['km'])
    y = dataset['price']

    metrics_dict = model.fit(X, y, epochs=args.max_epochs)
    model.save('model.json')

    if args.display_metric:
        for metric in metrics_dict:
            plt.title(metric)
            plt.plot(range(len(metrics_dict[metric])), metrics_dict[metric])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(f'{metric}.png')
    
    