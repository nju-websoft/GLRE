#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np

from data.dataset import DocRelationDataset
from data.loader import DataLoader, ConfigLoader
from nnet.trainer import Trainer
from utils.utils import setup_log, load_model, load_mappings


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)              # Numpy module
    random.seed(seed)                 # Python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(parameters):
    model_folder = setup_log(parameters, parameters['save_pred'] + '_train')
    set_seed(parameters['seed'])

    ###################################
    # Data Loading
    ###################################
    # if parameters['re_train']:
    #     print('\nLoading mappings ...')
    #     train_loader = load_mappings(parameters['remodelfile'])
    # else:
    print('Loading training data ...')
    train_loader = DataLoader(parameters['train_data'], parameters)
    train_loader(embeds=parameters['embeds'], parameters=parameters)
    train_data, _ = DocRelationDataset(train_loader, 'train', parameters, train_loader).__call__()

    print('\nLoading testing data ...')
    test_loader = DataLoader(parameters['test_data'], parameters, train_loader)
    test_loader(parameters=parameters)
    test_data, prune_recall = DocRelationDataset(test_loader, 'test', parameters, train_loader).__call__()

    # print("prune_recall-->", str(prune_recall))
    ###################################
    # Training
    ###################################
    trainer = Trainer(train_loader, parameters, {'train': train_data, 'test': test_data}, model_folder, prune_recall)

    trainer.run()

    # if parameters['plot']:
    #     plot_learning_curve(trainer, model_folder)

    # if parameters['save_model']:
    #     save_model(model_folder, trainer, train_loader)


def _test(parameters):
    model_folder = setup_log(parameters, parameters['save_pred'] + '_test')

    print('\nLoading mappings ...')
    train_loader = load_mappings(parameters['remodelfile'])
    
    print('\nLoading testing data ...')
    test_loader = DataLoader(parameters['test_data'], parameters, train_loader)
    test_loader(parameters=parameters)
    test_data, prune_recall = DocRelationDataset(test_loader, 'test', parameters, train_loader).__call__()

    m = Trainer(train_loader, parameters, {'train': [], 'test': test_data}, model_folder, prune_recall)
    trainer = load_model(parameters['remodelfile'], m)
    trainer.eval_epoch(final=True, save_predictions=True)


def main():
    config = ConfigLoader()
    parameters = config.load_config()

    if parameters['train']:
        train(parameters)

    elif parameters['test']:
        _test(parameters)

if __name__ == "__main__":
    main()

