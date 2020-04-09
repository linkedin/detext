import tensorflow as tf
from tensorflow.python.tools.saved_model_cli import _show_inputs_outputs

PLACEHOLDER_SUFFIX = '_placeholder'


def generate_sub_model(original_saved_model_dir, export_model_dir, input_tensor_names, output_tensor_names):
    """
    Modify original saved model to generate sub model in saved model format based on given input & output tensor names.
    Use this function to create "sub-graphs" such that pre-computing some nodes in the original graph could be enabled.
    Nodes that are not on the path of the "sub-graph" will not be included in the new graph.
    :param original_saved_model_dir: original saved model (*.pb) dir.
    :param export_model_dir: new dir to save the generated model. If exists, the dir will be removed first.
    :param input_tensor_names: inputs to include in new model separated by comma.
    Ensure the tensor names are in the computation graph.
    Eg. 'doc_currTitles_placeholder:0,doc_headlines_placeholder:0'
    :param output_tensor_names: outputs to include in the new model.
    Ensure the tensor names are in the computation graph. Eg. 'doc_ftrs:0'
    :return:
    """
    # Remove existing dir
    print('************** Generating sub model ********************')
    print('Inputs: ' + input_tensor_names)
    print('Outputs: ' + output_tensor_names)
    if tf.gfile.Exists(export_model_dir):
        print('Removing old output dir' + export_model_dir)
        tf.gfile.DeleteRecursively(export_model_dir)

    # Get list of input and output tensors
    input_tensor_list = input_tensor_names.split(',')
    output_tensor_list = output_tensor_names.split(',')

    with tf.Session(graph=tf.Graph()) as sess:
        # Load saved model graph
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], original_saved_model_dir)
        graph = tf.get_default_graph()

        # Get signature dict from list of input and output tensors
        inputs = get_signature_dict(graph, input_tensor_list)
        outputs = get_signature_dict(graph, output_tensor_list)

        # Save to converted_model_dir with updated inputs and outputs
        tf.saved_model.simple_save(
            sess,
            export_model_dir,
            inputs=inputs,
            outputs=outputs
        )
        print('New model saved! Location: ' + export_model_dir)
        print('New model signatures:')
        # Use saved_model_cli tool to show the model signatures
        _show_inputs_outputs(export_model_dir,
                             tf.saved_model.tag_constants.SERVING,
                             tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                             indent=0)


def get_signature_dict(graph, tensor_names_list, placeholder_suffix=PLACEHOLDER_SUFFIX):
    """
    Generate dict (inputs or outputs) for saved_model. The key is the name of the input placeholders or output tensors,
    and value is the tensors. input_name is the substring of the tensor name removing all after '_placeholder' to match
    the naming style in default model.
    :param graph: graph used to get tensors.
    :param tensor_names_list: tensors to put into the returned dict.
    :return: a dictionary that contains the <name: tensor> mapping
    """
    signature_dict = {}
    for t in tensor_names_list:
        input_name = t.split(placeholder_suffix)[0]
        signature_dict[input_name] = graph.get_tensor_by_name(t)
    return signature_dict


def generate_serving_models_ps_l1(orig_model_dir):
    """
    Generate the sub models for people search l1 models.
    This will generate:
    1. A query model for computing query embeddings online.
    2. A doc model for computing document embeddings to pre-compute all doc embeddings offline.
    3. A sim-wide model for using similarity scores and wide features to compute final scores online.
    :param orig_model_dir: original model directory
    :return:
    """
    print('Original model signature:')
    # Use saved_model_cli tool to show the model signatures
    _show_inputs_outputs(orig_model_dir,
                         tf.saved_model.tag_constants.SERVING,
                         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                         indent=0)

    # Query model for online inferencing query embeddings
    input_tensors = 'query_placeholder:0'
    outputs_tensors = 'query_ftrs:0'
    generate_sub_model(orig_model_dir, orig_model_dir + '-query-model', input_tensors, outputs_tensors)

    # Document model for pre-computing document embeddings
    input_tensors = 'doc_currTitles_placeholder:0,doc_headlines_placeholder:0,doc_prevTitles_placeholder:0'
    outputs_tensors = 'doc_ftrs:0'
    generate_sub_model(orig_model_dir, orig_model_dir + '-doc-model', input_tensors, outputs_tensors)

    # Online inference model to compute final scores using similarity features and wide features
    input_tensors = 'sim_ftrs:0,wide_ftr_placeholder:0'
    outputs_tensors = 'final_scores:0'
    generate_sub_model(orig_model_dir, orig_model_dir + '-sim-wide-model', input_tensors, outputs_tensors)


if __name__ == '__main__':
    generate_serving_models_ps_l1('/tmp/psmodel/best_ndcg@10')
