import importlib
import itertools
import json
import logging
import sys

import click
import numpy as np
import tensorflow as tf
from experiment.util import replace_dict_values

from experiment.config import load_config

# Allows the gpu to be used in parallel
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True


@click.command()
@click.argument('config_file')
def run(config_file):
    """This program allows to perform hyperparameter optimization with grid search and cross validation.

    """
    config = load_config(config_file)

    # to reproduce the results
    np.random.seed(1)
    tf.set_random_seed(1)

    # We are now fetching all relevant modules. It is strictly required that these module contain a variable named
    # 'component' that points to a class which inherits from experiment.Data, experiment.Experiment, experiment.Trainer
    # or experiment.Evaluator
    data_module = config['data-module']
    model_module = config['model-module']
    training_module = config['training-module']

    # The modules are now dynamically loaded
    DataClass = importlib.import_module(data_module).component
    ModelClass = importlib.import_module(model_module).component
    TrainingClass = importlib.import_module(training_module).component

    # setup a logger
    logger = logging.getLogger('experiment')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel(config['logger']['level'])
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_stdout)

    if 'path' in config['logger']:
        handler_file = logging.FileHandler(config['logger']['path'])
        handler_file.setLevel(config['logger']['level'])
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_file)

    logger.setLevel(config['logger']['level'])

    # We then wire together all the modules
    config_global = config['global']
    config_optimization = config['optimization']

    logger.info('Setting up the data')
    data_complete = DataClass(config['data'], config_global, logger)
    data_complete.setup()

    # Cross fold optimization
    # We first need to create all the configuration choices for grid search
    optimization_parameters = config_optimization['parameters']
    grid = list(itertools.product(*config_optimization['parameters'].values()))
    logger.info('We have {} different hyperparameter combinations'.format(len(grid)))

    # We now go over all choices and perform cross-fold validation
    avg_scores = []
    for configuration_values_overwrite in grid:
        parameter_choices = list(zip(optimization_parameters.keys(), configuration_values_overwrite))
        config_run = replace_dict_values(config, parameter_choices)

        logger.info('-' * 40)
        logger.info('Checking configuration {}'.format(json.dumps(parameter_choices)))

        # Run each fold
        n_folds = config_optimization['folds']
        scores = []
        for fold_i in range(n_folds):
            logger.info('Starting fold {}/{}'.format(fold_i + 1, n_folds))
            with tf.Session(config=sess_config) as sess:
                training = TrainingClass(config_run['training'], config_global, logger)
                model = ModelClass(config_run['model'], config_global, logger)
                model.build(data_complete, sess)
                data_fold = data_complete.get_fold_data(fold_i, n_folds)
                best_epoch, best_score = training.start(model, data_fold, sess)
                scores.append(best_score)
                logger.info('Fold {}/{} performance: {}'.format(fold_i + 1, n_folds, best_score))
                logger.info('-' * 20)
                training.remove_checkpoints()
            tf.reset_default_graph()

        logger.info('Training ended')
        logger.info('All scores: {}'.format(json.dumps(scores)))
        avg_score = np.mean(scores)
        logger.info('Avg score for current configuration: {}'.format(avg_score))
        avg_scores.append(avg_score)

    best_configuration = grid[np.argmax(avg_scores)]
    logger.info('Grid search completed')
    logger.info('Best configuration: {} (score={})'.format(best_configuration, max(avg_scores)))
    logger.info('-')
    logger.info('All configurations: {}'.format(json.dumps(grid)))
    logger.info('All scores: {}'.format(avg_scores))

    logger.info('DONE')


if __name__ == '__main__':
    run()
