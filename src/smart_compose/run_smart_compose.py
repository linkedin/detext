import sys
import tempfile
from dataclasses import asdict

import tensorflow as tf
from absl import logging
from official.utils.misc import distribution_utils

from smart_compose.args import SmartComposeArg
from smart_compose.train import train
from smart_compose.utils import distributed_utils, parsing_utils


def main(argv):
    """ This is the main method for training the model.

    :param argv: training parameters
    """

    argument = SmartComposeArg.__from_argv__(argv[1:], error_on_unknown=False)
    logging.set_verbosity(logging.INFO)
    logging.info(f"Args:\n {argument}")

    hparams = argument

    strategy = distribution_utils.get_distribution_strategy(hparams.distribution_strategy, num_gpus=hparams.num_gpu, all_reduce_alg=hparams.all_reduce_alg)
    logging.info(f"***********Num replica: {strategy.num_replicas_in_sync}***********")
    create_output_dir(hparams.resume_training, hparams.out_dir, strategy)

    save_hparams(hparams.out_dir, parsing_utils.HParams(**asdict(argument)), strategy)

    logging.info("***********Smart Compose Training***********")
    return train.train(strategy, hparams)


def save_hparams(out_dir, hparams, strategy):
    """Saves hparams to out_dir"""
    is_chief = distributed_utils.is_chief(strategy)
    if not is_chief:
        out_dir = tempfile.mkdtemp()

    parsing_utils.save_hparams(out_dir, hparams)

    if not is_chief:
        tf.io.gfile.remove(parsing_utils._get_hparam_path(out_dir))


def create_output_dir(resume_training, out_dir, strategy):
    """Creates output directory if not exists"""
    is_chief = distributed_utils.is_chief(strategy)
    if not is_chief:
        out_dir = tempfile.mkdtemp()

    if not resume_training:
        if tf.io.gfile.exists(out_dir):
            logging.info("Removing previous output directory...")
            tf.io.gfile.rmtree(out_dir)

    # If output directory deleted or does not exist, create the directory.
    if not tf.io.gfile.exists(out_dir):
        logging.info('Creating dirs recursively at: {0}'.format(out_dir))
        tf.io.gfile.makedirs(out_dir)

    if not is_chief:
        tf.io.gfile.rmtree(out_dir)


if __name__ == '__main__':
    main(sys.argv)
