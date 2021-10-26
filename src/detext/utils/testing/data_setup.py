import os

import tensorflow as tf

from detext.utils.parsing_utils import HParams
from detext.utils.vocab_utils import read_vocab


class DataSetup:
    """Class containing common setup on file paths, layer params used in unit tests"""
    resource_dir = os.path.join(os.getcwd(), 'test', 'detext', 'resources')
    we_file = os.path.join(resource_dir, 'we.pkl')
    vocab_file = os.path.join(resource_dir, 'vocab.txt')
    vocab_file_for_id_ftr = vocab_file

    vocab_layer_dir = os.path.join(resource_dir, 'vocab_layer')
    embedding_layer_dir = os.path.join(resource_dir, 'embedding_layer')

    bert_hub_url = os.path.join(resource_dir, 'bert-hub')
    libert_sp_hub_url = os.path.join(resource_dir, 'libert-sp-hub')
    libert_space_hub_url = os.path.join(resource_dir, 'libert-space-hub')
    vocab_hub_url = os.path.join(resource_dir, 'vocab_layer_hub')
    embedding_hub_url = os.path.join(resource_dir, 'embedding_layer_hub')

    out_dir = os.path.join(resource_dir, "output")
    data_dir = os.path.join(resource_dir, "train", "dataset", "tfrecord")
    multitask_data_dir = os.path.join(resource_dir, "train", "multitask", "tfrecord")
    cls_data_dir = os.path.join(resource_dir, "train", "classification", "tfrecord")
    binary_cls_data_dir = os.path.join(resource_dir, "train", "binary_classification", "tfrecord")
    ranking_data_dir = os.path.join(resource_dir, "train", "ranking", "tfrecord")

    vocab_table_py = read_vocab(vocab_file)
    vocab_size = len(vocab_table_py)

    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    UNK = '[UNK]'
    CLS_ID = vocab_table_py[CLS]
    PAD_ID = vocab_table_py[PAD]
    SEP_ID = vocab_table_py[SEP]
    UNK_ID = vocab_table_py[UNK]

    PAD_FOR_ID_FTR = PAD
    UNK_FOR_ID_FTR = UNK

    query = tf.constant(['batch1',
                         'batch 2 query build'], dtype=tf.dtypes.string)
    query_length = [1, 4]

    user_id_field1 = query
    user_id_field2 = query

    cls_doc_field1 = ['same content build', 'batch 2 field 1 word']
    cls_doc_field2 = ['same content build', 'batch 2 field 2 word']

    ranking_doc_field1 = [['same content build',
                           'batch 1 doc 2 field able',
                           'batch 1 doc 3 field 1'],
                          ['batch 2 doc 1 field word',
                           'batch 2 doc 2 field 1',
                           'batch 2 doc 3 field test']]
    ranking_doc_id_field1 = ranking_doc_field1

    ranking_doc_field2 = [['same content build',
                           'batch 1 doc 2 field test',
                           'batch 1 doc 3 field 2'],
                          ['batch 2 doc 1 field test',
                           'batch 2 doc 2 field 2',
                           'batch 2 doc 3 field word']]
    ranking_doc_id_field2 = ranking_doc_field2

    cls_sparse_features_1 = [[1.0, 2.0, 4.0], [0.0, -1.0, 4.0]]
    cls_sparse_features_2 = [[2.0, 0.0, 4.0], [2.0, 2.0, 4.0]]
    cls_sparse_features = [tf.sparse.from_dense(tf.constant(cls_sparse_features_1)),
                           tf.sparse.from_dense(tf.constant(cls_sparse_features_2))]

    ranking_sparse_features_1 = [[[1.0, 2.0, 4.0],
                                  [1.0, 2.0, 4.0],
                                  [1.0, 2.0, 4.0]],
                                 [[0.0, -1.0, 4.0],
                                  [0.0, -1.0, 4.0],
                                  [0.0, -1.0, 4.0]]]
    ranking_sparse_features_2 = [[[1.0, 2.0, 4.0],
                                  [1.0, 2.0, 4.0],
                                  [1.0, 2.0, 4.0]],
                                 [[0.0, -1.0, 4.0],
                                  [0.0, -1.0, 4.0],
                                  [0.0, -1.0, 4.0]]]
    ranking_sparse_features = [tf.sparse.from_dense(tf.constant(ranking_sparse_features_1)),
                               tf.sparse.from_dense(tf.constant(ranking_sparse_features_2))]
    nums_sparse_ftrs = [3]
    total_num_sparse_ftrs = sum(nums_sparse_ftrs)
    sparse_embedding_size = 33

    num_user_fields = 2
    user_fields = [tf.constant(query, dtype=tf.dtypes.string), tf.constant(query, dtype=tf.dtypes.string)]

    num_doc_fields = 2
    ranking_doc_fields = [tf.constant(ranking_doc_field1, dtype=tf.dtypes.string), tf.constant(ranking_doc_field2, dtype=tf.dtypes.string)]
    cls_doc_fields = [tf.constant(cls_doc_field1, dtype=tf.dtypes.string), tf.constant(cls_doc_field2, dtype=tf.dtypes.string)]

    num_user_id_fields = 2
    user_id_fields = user_fields

    num_doc_id_fields = 2
    ranking_doc_id_fields = ranking_doc_fields
    cls_doc_id_fields = cls_doc_fields

    num_id_fields = num_user_id_fields + num_doc_id_fields

    num_units = 6
    num_units_for_id_ftr = num_units

    vocab_layer_param = {'CLS': CLS,
                         'SEP': SEP,
                         'PAD': PAD,
                         'UNK': UNK,
                         'vocab_file': vocab_file}

    embedding_layer_param = {'vocab_layer_param': vocab_layer_param,
                             'vocab_hub_url': '',
                             'we_file': '',
                             'we_trainable': True,
                             'num_units': num_units}

    min_len = 3
    max_len = 7
    filter_window_sizes = [1, 2, 3]
    num_filters = 5

    cnn_param = HParams(
        filter_window_sizes=filter_window_sizes,
        num_filters=num_filters, num_doc_fields=num_doc_fields, num_user_fields=num_user_fields,
        min_len=min_len, max_len=max_len,
        embedding_layer_param=embedding_layer_param, embedding_hub_url=None)

    id_encoder_param = HParams(num_id_fields=num_id_fields,
                               embedding_layer_param=embedding_layer_param, embedding_hub_url_for_id_ftr=None)
    rep_layer_param = HParams(ftr_ext='cnn',
                              num_doc_fields=num_doc_fields, num_user_fields=num_user_fields,
                              num_doc_id_fields=num_doc_id_fields, num_user_id_fields=num_user_id_fields,
                              add_doc_projection=False, add_user_projection=False,
                              text_encoder_param=cnn_param, id_encoder_param=id_encoder_param)
