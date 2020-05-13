"""Utility functions related to distributed training executor setup."""
import json
import logging
import os
import tensorflow as tf

CHIEF = 'chief'
EVALUATOR = 'evaluator'
WORKER = 'worker'
PARAMETER_SERVER = 'ps'
LOCAL_MODE = 'local'


def get_executor_task_type():
    """Get executor task type (chief/evaluator/worker/ps or local) from TF_CONFIG."""
    tf.logging.info("Getting executor context info...")
    tf_config = os.environ.get("TF_CONFIG")

    # With multiworker training, tf_config contains tasks in the cluster, and each task's type in the cluster.
    # To get ther worker/evaluator status, need to fetch corresponding fields in the config (json format).
    # Read more at https://www.tensorflow.org/guide/distributed_training#setting_up_tf_config_environment_variable
    if tf_config:
        tf_config_json = json.loads(tf_config)

        # Logging the status of current worker/evaluator
        logging.info("Running with TF_CONFIG: {}".format(tf_config_json))
        task_type = tf_config_json.get('task', {}).get('type')
        logging.info("=========== Current executor task type: {} ==========".format(task_type))
        return task_type
    else:
        logging.info("=========== No TF_CONFIG found. Running {} mode. ==========".format(LOCAL_MODE))
        return LOCAL_MODE
