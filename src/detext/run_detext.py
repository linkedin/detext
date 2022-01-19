"""
Overall pipeline to train the model.  It parses arguments, and trains a DeText model.
"""

import tempfile
from dataclasses import dataclass, asdict

import sys
import tensorflow as tf
from absl import logging
from official.utils.misc import distribution_utils
from smart_arg import arg_suite

from detext.args import DatasetArg, FeatureArg, NetworkArg, OptimizationArg
from detext.train import train
from detext.utils import parsing_utils, distributed_utils


@arg_suite
@dataclass
class DetextArg(DatasetArg, FeatureArg, NetworkArg, OptimizationArg):
    """
    DeText: a Deep Text understanding framework for NLP related ranking, classification, and language generation tasks.

    It leverages semantic matching using deep neural networks to understand member intents in search and recommender systems.
    As a general NLP framework, currently DeText can be applied to many tasks, including search & recommendation ranking,
    multi-class classification and query understanding tasks.
    """

    def __post_init__(self):
        """ Post initializes fields

        This method is automatically called by smart-arg once the argument is created by parsing cli or the constructor
        """
        logging.info(f"Start __post_init__ the argument now: {self}")
        super().__post_init__()


def main(argv):
    """ This is the main method for training the model.

    :param argv: training parameters
    """

    argument = DetextArg.__from_argv__(argv[1:], error_on_unknown=False)
    run_detext(argument)


def run_detext(argument):
    """ Launches DeText training program"""
    logging.set_verbosity(logging.INFO)
    logging.info(f"Args:\n {argument}")

    hparams = parsing_utils.HParams(**asdict(argument))

    strategy = distribution_utils.get_distribution_strategy(hparams.distribution_strategy, num_gpus=hparams.num_gpu, all_reduce_alg=hparams.all_reduce_alg)
    logging.info(f"***********Num replica: {strategy.num_replicas_in_sync}***********")
    create_output_dir(hparams.resume_training, hparams.out_dir, strategy)
    save_hparams(hparams.out_dir, hparams, strategy)

    logging.info("***********DeText Training***********")
    train.train(strategy, hparams)


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
