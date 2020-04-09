import tensorflow as tf
import numpy as np
import pickle
import sys
import random
from avro.datafile import DataFileReader
from avro.io import DatumReader


MAX_FILE = 50
random.seed(4321)


def get_mean_std(input_files_patten, norm_file, num_wide):
    """Compute mean and std"""
    file_format = input_files_patten.split('.')[-1]
    all_ftrs = [[] for _ in range(num_wide)]
    tf.logging.set_verbosity(tf.logging.INFO)
    # read data, apply normalization on wide features
    files = tf.gfile.Glob(input_files_patten)
    random.shuffle(files)
    if file_format == 'tfrecords':
        for input_file in files[:MAX_FILE]:
            tf.logging.info(input_file)
            for example in tf.python_io.tf_record_iterator(input_file):
                result = tf.train.Example.FromString(example)
                wide_ftrs = result.features.feature['wide_ftrs'].float_list.value
                for i, f in enumerate(wide_ftrs):
                    i2 = i % num_wide
                    all_ftrs[i2].append(f)
    elif file_format == 'avro':
        for input_file in files[:MAX_FILE]:
            tf.logging.info(input_file)
            reader = DataFileReader(tf.gfile.Open(input_file, "rb"), DatumReader())
            for record in reader:
                wide_ftrs = record['wide_ftrs']
                for i, f in enumerate(wide_ftrs):
                    i2 = i % num_wide
                    all_ftrs[i2].append(f)
    else:
        raise ValueError("Unsupported file format %s." % file_format)
    mean_value = [0] * num_wide
    std_value = [0] * num_wide
    for i in range(num_wide):
        tf.logging.info('%d %f %f' % (i, np.mean(all_ftrs[i]), np.std(all_ftrs[i])))
        mean_value[i] = np.mean(all_ftrs[i]).item()
        std_value[i] = np.std(all_ftrs[i]).item()
    with tf.gfile.Open(norm_file, 'wb') as fout:
        pickle.dump((mean_value, std_value), fout, protocol=2)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage:  python  get_mean_std.py  input_files_patten  output_file  num_wide')
        sys.exit()
    input_files_patten = sys.argv[1]
    output_file = sys.argv[2]
    num_wide = int(sys.argv[3])
    get_mean_std(input_files_patten, output_file, num_wide)
