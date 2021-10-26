"""Unit test utilities"""
import pickle
import time

import numpy as np
import tensorflow as tf


def create_sample_tfrecord(out_file):
    """Creates sample tfrecord to out_file"""
    print("Removing existing file {}".format(out_file))
    if tf.io.gfile.exists(out_file):
        tf.io.gfile.remove(out_file)

    target_column_name = 'query'
    input_column_name = 'user_input'

    target_cases = [b'word function', b'data', b'', b'this is a sentence']
    input_cases = [b'word function', b'data', b'', b'this is a sentence']

    print("Composing fake tfrecord to file {}".format(out_file))
    with tf.io.TFRecordWriter(out_file) as writer:
        with tf.Graph().as_default(), tf.compat.v1.Session():
            num_instances_per_case = 10

            target_list = target_cases * num_instances_per_case
            input_list = input_cases * num_instances_per_case

            for inp, target in zip(input_list, target_list):
                features = {
                    input_column_name: _bytes_feature([inp]),
                    target_column_name: _bytes_feature([target]),
                }
                example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example_proto.SerializeToString())


def _bytes_feature(value):
    """Returns a bytes_list feature"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list feature"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list feature"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def make_we_file(vocab_file, embedding_size, output_path):
    """Generates a word embedding file to output_path """
    embedding = []
    with tf.io.gfile.GFile(vocab_file, 'r') as fin:
        for _ in fin:
            embedding.append(np.random.uniform(-1, 1, [embedding_size]))
    embedding = np.array(embedding)
    pickle.dump(embedding, tf.io.gfile.GFile(output_path, 'w'))
    print(f'Dumped embedding to {output_path}')


def timeit(method):
    """A decorator function to measure the run time of a function"""

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


if __name__ == '__main__':
    from smart_compose.utils.testing.data_setup import DataSetup
    from os.path import join

    create_sample_tfrecord(join(DataSetup.data_dir, 'test.tfrecord'))
