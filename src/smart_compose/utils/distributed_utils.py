""" Distributed training utilities """
import json
import os

import tensorflow as tf
from absl import logging


def should_export_summary(strategy):
    """Returns whether the summary should be exported given current strategy"""
    return (not strategy) or strategy.extended.should_save_summary


def should_export_checkpoint(strategy):
    """Returns whether the checkpoint should be exported given current strategy"""
    return (not strategy) or strategy.extended.should_checkpoint


def is_chief(strategy):
    """Returns whether the current node is chief node"""
    tf_config = os.environ.get("TF_CONFIG")
    # With multiworker training, tf_config contains tasks in the cluster, and each task's type in the cluster.
    # To get ther worker/evaluator status, need to fetch corresponding fields in the config (json format).
    # Read more at https://www.tensorflow.org/guide/distributed_training#setting_up_tf_config_environment_variable
    if tf_config:
        tf_config_json = json.loads(tf_config)
        # Logging the status of current worker/evaluator
        logging.info("Running with TF_CONFIG: {}".format(tf_config_json))
        task = tf_config_json.get('task', {})
        task_type = task.get('type', None)
        task_id = task.get('index', None)
        logging.info(f"=========== Current executor task type: {task_type}, task id: {task_id} ==========")
        return _is_chief(task_type, task_id, strategy)
    else:
        logging.info("=========== No TF_CONFIG found. Running local mode. ==========")
        return True


def _is_chief(task_type, task_id, strategy):
    # If `task_type` is None, this may be operating as single worker, which works
    #   effectively as chief.
    if task_type is None:
        return True
    if isinstance(strategy, tf.distribute.experimental.MultiWorkerMirroredStrategy):
        return task_type == 'worker' and task_id == 0
    return task_type == 'chief'
