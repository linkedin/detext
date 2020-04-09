import codecs
import json
import numpy as np
import os
import pickle
import random
import tensorflow as tf

from detext.model.bert import modeling
from detext.utils import test_utils
from detext.utils import vocab_utils


def force_set_hparam(hparams, name, value):
    """
    Removes name from hparams and sets hparams.name == value.
    This function is introduced because hparams.set_hparam(name, value) requires value to be of the same type as the
        existing hparam.get(name) if name is already set in hparam
    """
    hparams.del_hparam(name)
    hparams.add_hparam(name, value)


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True
    return config_proto


def save_hparams(out_dir, hparams):
    """Save hparams."""
    hparams_file = os.path.join(out_dir, "hparams")
    print("  saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json())


def load_hparams(model_dir):
    """Load hparams from an existing model directory."""
    hparams_file = os.path.join(model_dir, "hparams")
    if tf.gfile.Exists(hparams_file):
        print("# Loading hparams from %s" % hparams_file)
        with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
            try:
                hparams_values = json.load(f)
                hparams = tf.contrib.training.HParams(**hparams_values)
            except ValueError:
                print("  can't load hparams file")
                return None
        return hparams
    else:
        return None


def extend_hparams(hparams):
    # Sanity check for RNN related hparams
    assert hparams.unit_type in ['lstm', 'gru', 'layer_norm_lstm'], 'Only support lstm/gru/layer_norm_lstm as unit_type'
    assert hparams.num_layers > 0, 'num_layers must be larger than 0'
    assert hparams.num_residual_layers >= 0, 'num_residual_layers must >= 0'
    assert 0 <= hparams.forget_bias <= 1, 'forget_bias must be within [0.0, 1.0]'
    assert 0 <= hparams.rnn_dropout <= 1, 'rnn_dropout must be within [0.0, 1.0]'

    # Get number of doc/usr text fields
    num_doc_fields = sum([name.startswith('doc_') for name in hparams.feature_names.split(',')])
    hparams.add_hparam("num_doc_fields", num_doc_fields)
    num_usr_fields = sum([name.startswith('usr_') for name in hparams.feature_names.split(',')])
    hparams.add_hparam("num_usr_fields", num_usr_fields)

    # Get number of doc/usr id fields
    num_doc_id_fields = sum([name.startswith('docId_') for name in hparams.feature_names.split(',')])
    hparams.add_hparam("num_doc_id_fields", num_doc_id_fields)
    num_usr_id_fields = sum([name.startswith('usrId_') for name in hparams.feature_names.split(',')])
    hparams.add_hparam("num_usr_id_fields", num_usr_id_fields)
    if num_doc_id_fields > 0 or num_usr_id_fields > 0:
        assert hparams.vocab_file_for_id_ftr is not None, \
            "Must provide vocab_field_for_id_ftr arg when id features are provided"

    # find vocab size, pad id from vocab file
    vocab_table = vocab_utils.read_vocab(hparams.vocab_file)
    hparams.add_hparam("vocab_size", len(vocab_table))
    hparams.pad_id = vocab_table[hparams.PAD]

    # find vocab size, pad id from vocab file for id features
    if hparams.vocab_file_for_id_ftr is not None:
        vocab_table_for_id_ftr = vocab_utils.read_vocab(hparams.vocab_file_for_id_ftr)
        hparams.add_hparam("vocab_size_for_id_ftr", len(vocab_table_for_id_ftr))
        hparams.pad_id_for_id_ftr = vocab_table_for_id_ftr[hparams.PAD_FOR_ID_FTR]

    # if there is bert config, check compatibility of between bert parameters and existing parameters
    if hparams.bert_config_file:
        hparams.bert_config = modeling.BertConfig.from_json_file(hparams.bert_config_file)
        assert hparams.bert_config.vocab_size == hparams.vocab_size

    # The regex pattern to add a white space before and after. Used for processing text fields.
    tok2regex_pattern = {'plain': None, 'punct': r'(\pP)'}
    hparams.regex_replace_pattern = tok2regex_pattern[hparams.tokenization]

    # if bert, then disable cnn parameters
    if hparams.ftr_ext != 'cnn':
        hparams.filter_window_sizes = '0'

    # convert from string to arrays for filter_window_sizes
    filter_window_sizes_str = hparams.filter_window_sizes
    force_set_hparam(hparams, "filter_window_sizes", [int(x.strip()) for x in filter_window_sizes_str.split(',')])

    assert hparams.pmetric is not None, "Please set your primary evaluation metric using --pmetric option"
    assert hparams.pmetric != 'confusion_matrix', 'confusion_matrix cannot be used as primary evaluation metric.'

    # Set all relevant evaluation metrics
    all_metrics = hparams.all_metrics.split(',') if hparams.all_metrics else [hparams.pmetric]
    assert hparams.pmetric in all_metrics, "pmetric must be within all_metrics"
    force_set_hparam(hparams, "all_metrics", all_metrics)

    # convert from string to arrays for num_hidden
    num_hidden_str = str(hparams.num_hidden)
    force_set_hparam(hparams, "num_hidden", [int(x.strip()) for x in num_hidden_str.split(',')])
    # convert from string to arrays for feature names
    setattr(hparams, 'feature_names', tuple(hparams.feature_names.split(',')))
    # lambda rank
    if hparams.lambda_metric is not None and hparams.lambda_metric == 'ndcg':
        setattr(hparams, 'lambda_metric', {'metric': 'ndcg', 'topk': 10})
    else:
        setattr(hparams, 'lambda_metric', None)
    # feature normalization
    if hparams.std_file:
        # read normalization file
        print('read normalization file')
        ftr_mean, ftr_std = _load_ftr_mean_std(hparams.std_file)
        hparams.add_hparam('ftr_mean', np.array(ftr_mean, dtype=np.float32))
        hparams.add_hparam('ftr_std', np.array(ftr_std, dtype=np.float32))

    # for score rescaling, the score_rescale has the xgboost mean and std.
    if hparams.score_rescale:
        force_set_hparam(hparams, 'score_rescale', [float(x) for x in hparams.score_rescale.split(',')])

    if hparams.explicit_empty:
        assert hparams.ftr_ext == 'cnn', 'explicit_empty will only be True when ftr_ext is cnn'

    # Convert string to arrays for emb_sim_func
    force_set_hparam(hparams, "emb_sim_func", hparams.emb_sim_func.split(','))

    # Checking hparam keep_checkpoint_max: must be >= 0
    if hparams.keep_checkpoint_max:
        assert hparams.keep_checkpoint_max >= 0

    # Classification task
    if hparams.num_classes > 1:
        # For classification tasks, restrict pmetric to be accuracy and use accuracy and confusion_matrix as metrics.
        hparams.pmetric = 'accuracy'
        hparams.all_metrics = ['accuracy', 'confusion_matrix']

    # L1 and L2 scale must be non-negative values
    assert hparams.l1 is None or hparams.l1 >= 0, "l1 scale must be non-negative"
    assert hparams.l2 is None or hparams.l2 >= 0, "l1 scale must be non-negative"
    return hparams


def clean_tfreocrds(input_file, output_file):
    """Clean tfrecords file"""
    writer = tf.python_io.TFRecordWriter(output_file)
    count = 0
    for example in tf.python_io.tf_record_iterator(input_file):
        result = tf.train.Example.FromString(example)
        labels = result.features.feature['label'].float_list.value
        # the labels should have at least 2 different values
        if len(set(labels)) != 1:
            writer.write(example)
        else:
            print(labels)
            count += 1
    print(str(count) + ' examples has the same labels')
    writer.close()


def shuffle_tfrecords(input_file, output_file):
    """shuffle tfrecords file"""
    writer = tf.python_io.TFRecordWriter(output_file)
    data = []
    for example in tf.python_io.tf_record_iterator(input_file):
        data.append(example)

    random.shuffle(data)
    for example in data:
        writer.write(example)
    writer.close()


def sample_tfrecords(input_file, sample_cnt, output_file):
    """sample tfrecords file"""
    writer = tf.python_io.TFRecordWriter(output_file)
    data = []
    for example in tf.python_io.tf_record_iterator(input_file):
        data.append(example)

    random.shuffle(data)
    cnt = 0
    for example in data:
        writer.write(example)
        cnt += 1
        if cnt == sample_cnt:
            break
    writer.close()


def random_baseline(input_files, topk):
    """compute random baseline NDCG"""
    if type(input_files) is not list:
        input_files = [input_files]
    ndcg_scores = []
    count = 0
    for input_file in input_files:
        for example in tf.python_io.tf_record_iterator(input_file):
            count += 1
            result = tf.train.Example.FromString(example)
            label = result.features.feature['label'].float_list.value
            for _ in range(1):
                scores = random.sample(range(len(label)), len(label))
                ndcg_scores.append(test_utils.get_ndcg(scores, label, topk))
    print(count)
    print("{} : {}".format("Random baseline NDCG", np.mean(ndcg_scores)))


def generate_latency_test_data(input_file, output_file, field_names, target_docs, num_wide):
    """
    Generate the data for latency test.
    For example, one query has 100 documents.
    """
    count = 0
    records = []
    # read data
    for example in tf.python_io.tf_record_iterator(input_file):
        result = tf.train.Example.FromString(example)
        records.append(result)
        count += 1
        # at most 1000 queries
        if count == 1000:
            break
    print('read ' + str(count), "records")

    # for each query
    with open(output_file, 'w') as fout:
        for i in range(len(records)):
            # query
            query = records[i].features.feature['query'].bytes_list.value[0].decode('utf-8')
            fout.write(query)
            fields = [[] for _ in field_names]
            j = 0
            while j < target_docs:
                index = random.randint(0, len(records) - 1)
                result = records[index]
                for k, field_name in enumerate(field_names):
                    if field_name == 'wide_ftrs':
                        ftrs = result.features.feature['wide_ftrs'].float_list.value
                    else:
                        ftrs = result.features.feature[field_name].bytes_list.value
                        ftrs = [x.decode('utf-8') for x in ftrs]
                    fields[k].extend(ftrs)
                j += len(result.features.feature['wide_ftrs'].float_list.value) / num_wide
            # wide features
            wide_ftrs_str = ' '.join(str(x) for x in fields[0][:target_docs * num_wide])
            fout.write('\t' + wide_ftrs_str)
            for field in fields[1:]:
                field_text = '**weiweiguo**'.join(field[:target_docs])
                fout.write('\t' + field_text)
            fout.write('\n')


def data_stats(input_files):
    """compute data statistics, such as click per result"""
    if type(input_files) is not list:
        input_files = [input_files]
    click_per_search = []
    for input_file in input_files:
        for example in tf.python_io.tf_record_iterator(input_file):
            result = tf.train.Example.FromString(example)
            label = result.features.feature['label'].float_list.value
            click_per_search.append(np.mean(label))
    print("{} : {}".format("Clicks per result", np.mean(click_per_search)))


def generate_unit_test_query_embeddings(input_file, savedmodel_dir, output_file, num_samples):
    """
    Generating query embeddings for online unit test
    :param input_file: tfrecord data containing query field
    :param savedmodel_dir: savedmodel path
    :param output_file: file name to embeddings
    :param num_samples: limit number of sample from input_file to write
    :return:
    """
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], savedmodel_dir)
        query_placeholder = tf.get_default_graph().get_tensor_by_name('query_placeholder:0')
        query_embedding = tf.get_default_graph().get_tensor_by_name('query_ftrs:0')

        query_list = []
        query_embedding_list = []
        for example in tf.python_io.tf_record_iterator(input_file):
            result = tf.train.Example.FromString(example)
            query = result.features.feature['query'].bytes_list.value[0].decode('utf-8')
            query_list.append(query)
            query_embeddings_v = sess.run(
                query_embedding,
                feed_dict={
                    query_placeholder: [query],
                }
            )
            query_embedding_list.append(query_embeddings_v[0])
            if len(query_embedding_list) == num_samples:
                break

        with open(output_file, 'w') as fout:
            for q, qe in zip(query_list, query_embedding_list):
                fout.write(q + ',' + ','.join(str(e) for e in qe))
                fout.write('\n')


def get_input_files(input_patterns):
    """Returns a list of file paths that match every pattern in input_patterns

    :param input_patterns a comma-separated string
    :return list of file paths
    """
    input_files = []
    for input_pattern in input_patterns.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))
    print("*** Input Files *** {} {}".format(input_patterns, len(input_files)))
    return input_files


def _load_ftr_mean_std(path):
    """ Loads mean and standard deviation from given file """
    with tf.gfile.Open(path, 'rb') as fin:
        if path.endswith("fromspark"):
            data = fin.readlines()
            # Line 0 is printing message, line 1 is feature mean, line 2 is feature std
            ftr_mean = [float(x.strip()) for x in data[1].decode("utf-8").split(',')]
            ftr_std = [float(x.strip()) for x in data[2].decode("utf-8").split(',')]
        else:
            ftr_mean, ftr_std = pickle.load(fin)
    # Put std val 0 -> 1 to avoid zero division error
    for i in range(len(ftr_std)):
        if ftr_std[i] == 0:
            ftr_std[i] = 1
    return ftr_mean, ftr_std


def estimate_train_steps(input_pattern, num_epochs, batch_size, isTfrecord):
    """
    Estimate train steps from number of epochs.
    Counting exact total nubmer of examples is time consuming and unnecessary,
    we count the first file and use the total file size to estimate
    total nubmer of examples.
    """
    # TODO: for now throw error if input file is avro format
    estimated_num_examples = 0
    tf.logging.info("Estimating train steps of {} epochs".format(num_epochs))
    if not isTfrecord:
        raise ValueError("--num_epochs doesn't support avro yet.")
    else:
        input_files = tf.gfile.Glob(input_pattern)

        file_1st = input_files[0]
        file_1st_num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(file_1st))
        tf.logging.info("number of examples in first file: {0}".format(file_1st_num_examples))

        file_1st_size = tf.gfile.GFile(file_1st).size()
        tf.logging.info("first file size: {0}".format(file_1st_size))

        file_size_num_example_ratio = float(file_1st_size) / file_1st_num_examples

        estimated_num_examples = sum([int(tf.gfile.GFile(fn).size() / file_size_num_example_ratio)
                                      for fn in input_files])

    tf.logging.info("Estimated number of examples: {0}".format(estimated_num_examples))

    num_train_steps = int(estimated_num_examples * num_epochs / batch_size)
    tf.logging.info("{0} epochs approximately need {1} train steps with batch size {2}".format(
        num_epochs,
        num_train_steps,
        batch_size))

    return num_train_steps


if __name__ == '__main__':
    # input_dir = '/home/wguo/projects/bert/people_v2/train_1m/'
    # output_file = '/home/wguo/projects/bert/people_v2/mean_std.pkl'
    # input_files = []
    # for file in sorted(os.listdir(input_dir)):
    #     input_files.append(os.path.join(input_dir, file))
    #
    # print(input_files[-2:])

    # input_files = [
    #     '/home/wguo/data/people_v6/NAV/train/00000',
    #     '/home/wguo/data/people_v6/NAV/train/00001',
    #     '/home/wguo/data/people_v6/NAV/train/00002',
    #     '/home/wguo/data/people_v6/NAV/train/00003',
    #     '/home/wguo/data/people_v6/NAV/train/00004',
    #     '/home/wguo/data/people_v6/NAV/train/00005',
    #     '/home/wguo/data/people_v6/NAV/train/00006',
    #     '/home/wguo/data/people_v6/NAV/train/00007',
    # ]
    # output_file = '/home/wguo/data/people_v6/NAV/mean_std.pkl'
    # get_mean_std(input_files, output_file, 167)
    # # shuffle_tfrecords(input_file, output_file)
    # # random_baseline('/home/xwli/ranking/job-search-data/dev/gen-tfrecords/output1.tfrecords', 25)
    #
    # # for people search latency test
    # input_file = ''
    # output_file = ''
    # field_names = ['wide_ftrs', 'doc_headlines', 'doc_currCompanies', 'doc_prevCompanies', 'doc_currTitles', 'doc_prevTitles', 'doc_schools']
    # target_docs = 200
    # generate_latency_test_data(input_file, output_file, field_names, target_docs)

    # # for job search latency test
    # input_file = '/home/wguo/projects/bert/job/dev.tfrecords'
    # output_file = '/home/wguo/projects/bert/job/dev.fake'
    # target_docs = 1000
    # num_wide = 96
    # field_names = ['wide_ftrs', 'doc_job_company', 'doc_job_title']
    # generate_latency_test_data(input_file, output_file, field_names, target_docs, num_wide)

    # random baseline
    # input_files = [
    #     '/home/wguo/data/people_v6/NAV/test/00000',
    #     '/home/wguo/data/people_v6/NAV/test/00001',
    #     '/home/wguo/data/people_v6/NAV/test/00002',
    #     '/home/wguo/data/people_v6/NAV/test/00003',
    #     '/home/wguo/data/people_v6/NAV/test/00004',
    # ]
    # random_baseline(input_files, 10)

    # input_files = [
    #     '/home/wguo/projects/bert/people_v4/00006',
    #     # '/home/wguo/projects/bert/people_v4/00001',
    # ]
    # data_stats(input_files)

    # for job search latency test
    # input_file = '/tmp/jsdata/dev-00000.tfrecords'
    # output_file = '/tmp/jsdata/dev-00000.tfrecords.1000'
    # target_docs = 1000
    # num_wide = 95
    # field_names = ['wide_ftrs', 'doc_job_company', 'doc_job_title']
    # generate_latency_test_data(input_file, output_file, field_names, target_docs, num_wide)

    # for people search latency test
    # input_file = '/home/xwli/ps-data/output-00000.tfrecords'
    # output_file = '/home/xwli/ps-data/output-00000.tfrecords.200'
    # target_docs = 200
    # num_wide = 167
    # field_names = ['wide_ftrs', 'doc_headlines', 'doc_currCompanies', 'doc_currTitles', 'doc_prevCompanies', 'doc_prevTitles']
    # generate_latency_test_data(input_file, output_file, field_names, target_docs, num_wide)

    # sample 1 tfrecord
    # input_file = '/tmp/psdata/L2/output-00003.tfrecords'
    # output_file = '/tmp/psdata/L2/sample1.tfrecords'
    # sample_tfrecords(input_file, 1, output_file)
    input_file = '/tmp/psdata/L2/test/output-00000.tfrecords'
    output_file = '/tmp/psdata/L2/query_embedding_samples.32d.test'
    num_samples = 20
    model_base_dir = '/home/xwli/models/ps-models/l2/expts_v2/ps-LiBERT-3-layer-32d-5df-100h-TrueDp-0.3TrainData-1'
    query_embedding_model = model_base_dir + '/savedmodel_query_embedding_only'
    generate_unit_test_query_embeddings(input_file, query_embedding_model, output_file, num_samples)
