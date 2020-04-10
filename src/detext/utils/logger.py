import tensorflow as tf


class Logger(object):
    """
    Logs the printed messages also to a log file. This redirect the printed info to stderr (for faster print out)
    and the log_file
    """

    def __init__(self, log_file):
        """

        :param log_file: The log_file filename for directing the messages to. Could be a local file or an hdfs file
        """
        tf.logging.set_verbosity(tf.logging.INFO)
        self.tf_logging = tf.logging
        self.writer = tf.gfile.Open(log_file, 'w')

    def write(self, message):
        """
        Writes to stderr, and log_file

        :param message: The message to write
        :return:
        """
        if message and message != '\n' and (not message.isspace()):
            self.tf_logging.info(message)
        self.writer.write(message)

    def flush(self):
        """

        This flush method is needed for python 3 compatibility.
        This handles the flush command by doing nothing.
        :return:
        """
        pass
