import tensorflow as tf

from detext.model.bert import modeling


def create_bert_model(bert_config_path,
                      is_training,
                      input_ids,
                      bert_checkpoint_path=None,
                      input_mask=None,
                      token_type_ids=None,
                      use_one_hot_embeddings=False
                      ):
    """
    Create a Bert model instance, with the option to initialize the params from a pretrained checkpoint.
    :param bert_config_path: config path of Bert model.
    :param bert_checkpoint_path: checkpoint path for pretrained model.
    :param is_training: true for training model, false for eval model. Controls whether dropout will be applied.
    :param input_ids: int32 Tensor of shape [batch_size, seq_length].
    :param input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
    :param token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length]. token_type_ids: (optional) int32
        Tensor of shape [batch_size, seq_length].
    :param use_one_hot_embeddings: (optional) bool. Whether to use one-hot word embeddings or tf.embedding_lookup()
        for the word embeddings. On the TPU, it is much faster if this is True, on the CPU or GPU, it is faster if this
        is False.
    :return: A Bert model instance.
    """
    if not bert_config_path:
        raise ValueError("A BERT config file must be given.")
    bert_config = modeling.BertConfig.from_json_file(bert_config_path)

    bert_model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=token_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )

    if bert_checkpoint_path:
        init_from_checkpoint(bert_checkpoint_path)

    return bert_model


def init_from_checkpoint(bert_checkpoint_path):
    """
    Given a checkpoint path of a pretrained Bert model, initialize the variables.
    :param bert_checkpoint_path: pretrained model checkpoint path.
    :return:
    """
    if not bert_checkpoint_path:
        raise ValueError("A BERT checkpoint path must be given to initialize the variables.")
    tvars = tf.trainable_variables()
    assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                             bert_checkpoint_path)
    tf.train.init_from_checkpoint(bert_checkpoint_path, assignment_map)
    print('%d variables are initialized from bert pretrained model' % len(initialized_variable_names))
