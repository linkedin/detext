import os
import shutil

import tensorflow as tf

from smart_compose.train import model
from smart_compose.utils.parsing_utils import InputFtrType, InternalFtrType
from smart_compose.utils.testing.test_case import TestCase
from smart_compose.utils.testing.testing_utils import timeit


class TestModel(TestCase):
    """Tests model.py"""
    inputs = {
        InputFtrType.TARGET_COLUMN_NAME:
            tf.constant(['this is a test sentence with 8 words'], dtype=tf.dtypes.string)
    }
    feature_type_2_name = {InputFtrType.TARGET_COLUMN_NAME: InputFtrType.TARGET_COLUMN_NAME}

    def testLatency(self):
        """Tests create_smart_compose_model() """
        inputs = self.inputs
        model_dir = os.path.join(self.resource_dir, 'tmp_model')
        if tf.io.gfile.exists(model_dir):
            tf.io.gfile.rmtree(model_dir)

        # Test model training output
        smart_compose_model = model.create_smart_compose_model(self.embedding_layer_with_large_vocab_layer_param, self.empty_url, self.min_len, self.max_len,
                                                               self.beam_width, self.max_decode_length, self.feature_type_2_name, self.min_seq_prob,
                                                               self.length_norm_power)

        outputs = smart_compose_model(inputs)
        smart_compose_model(inputs)
        smart_compose_model.get_training_probs_and_labels(inputs)

        # Test model export and loading
        smart_compose_model.save(model_dir)
        loaded_model = tf.keras.models.load_model(model_dir)

        @timeit
        def run_curr_model_prefix_aware_beam_search():
            num_runs = 100
            for i in range(num_runs):
                some_inputs = {
                    InputFtrType.TARGET_COLUMN_NAME:
                        tf.constant([f'this is a test sentence with {i} word'], dtype=tf.dtypes.string)
                }
                smart_compose_model.prefix_aware_beam_search(some_inputs)

        @timeit
        def run_curr_model_inference():
            num_runs = 100
            for i in range(num_runs):
                some_inputs = {
                    InputFtrType.TARGET_COLUMN_NAME:
                        tf.constant([f'this is a test sentence with {i} word'], dtype=tf.dtypes.string)
                }
                smart_compose_model(some_inputs)

        @timeit
        def run_loaded_model_inference():
            num_runs = 100
            for i in range(num_runs):
                some_inputs = {
                    InputFtrType.TARGET_COLUMN_NAME:
                        tf.constant([f'this is a test sentence with {i} word'], dtype=tf.dtypes.string)
                }
                loaded_model(some_inputs)

        run_curr_model_inference()  # ~11200 ms. No clue for now why this is slower. Will investigate in the future
        run_curr_model_prefix_aware_beam_search()  # ~1600 ms
        run_loaded_model_inference()  # ~2600 ms

        loaded_model_outputs = loaded_model(inputs)
        self.assertDictAllEqual(outputs, loaded_model_outputs)

        shutil.rmtree(model_dir)

    def testShape(self):
        """Tests LanguageModelingLayer """
        sentences = {
            InputFtrType.TARGET_COLUMN_NAME: tf.constant(['hello sent1', 'build'])
        }
        inputs = sentences
        batch_size = len(sentences[InputFtrType.TARGET_COLUMN_NAME])
        beam_width = 3
        max_decode_length = 2

        smart_compose_model = model.create_smart_compose_model(self.embedding_layer_param, self.empty_url, self.min_len, self.max_len,
                                                               beam_width, max_decode_length, self.feature_type_2_name, self.min_seq_prob,
                                                               self.length_norm_power)
        outputs = smart_compose_model.get_training_probs_and_labels(inputs)

        complete_sent_len = tf.size(sentences[InputFtrType.TARGET_COLUMN_NAME]) + smart_compose_model.num_cls_training + smart_compose_model.num_sep_training
        self.assertAllEqual(tf.shape(outputs[InternalFtrType.RNN_OUTPUT]),
                            [batch_size, complete_sent_len, self.vocab_size])
        self.assertAllEqual(tf.shape(outputs[InternalFtrType.SAMPLE_ID]), [batch_size, complete_sent_len])

        labels = outputs[InternalFtrType.LABEL]
        self.assertAllEqual(labels, [[0, 0, 2],
                                     [4, 2, 3]])

        logits = outputs[InternalFtrType.LOGIT]
        self.assertAllEqual(tf.shape(logits), [batch_size, 3, self.vocab_size])

        length = outputs[InternalFtrType.LENGTH]
        self.assertAllEqual(length, [3, 2])

    def testSampleOutput(self):
        """Tests language modeling layer and print sample outputs for given prefixes"""
        beam_width = 3
        max_decode_length = 2

        smart_compose_model = model.create_smart_compose_model(self.embedding_layer_param, self.empty_url, self.min_len, self.max_len,
                                                               beam_width, max_decode_length, self.feature_type_2_name, self.min_seq_prob,
                                                               self.length_norm_power)

        # {'exist_prefix': True,
        # 'predicted_scores': [[-2.7357671, -2.7361841, -2.7503903]] (could vary due to random initialization),
        # 'predicted_texts': [[b'[CLS] build is', b'[CLS] build source', b'[CLS] build token']]}
        print(smart_compose_model.prefix_aware_beam_search({
            InputFtrType.TARGET_COLUMN_NAME: tf.constant(['bui'])
        }))
        # {'exist_prefix': True,
        # 'predicted_scores': [[-2.7357671, -2.7361841, -2.7503903]] (could vary due to random initialization),
        # 'predicted_texts': [[b'[CLS] build is', b'[CLS] build source', b'[CLS] build token']]}
        print(smart_compose_model.prefix_aware_beam_search({
            InputFtrType.TARGET_COLUMN_NAME: tf.constant(['build'])
        }))
        # {'exist_prefix': True,
        # 'predicted_scores': [[-2.7357671, -2.7361841, -2.7503903]] (could vary due to random initialization),
        # 'predicted_texts': [[b'build is [PAD]', b'build source [PAD]', b'build token [PAD]']]}
        print(smart_compose_model.prefix_aware_beam_search({
            InputFtrType.TARGET_COLUMN_NAME: tf.constant(['build '])
        }))
        # {'exist_prefix': True,
        # 'predicted_scores': [[-2.711434 , -2.7171993, -2.7329462]] (could vary due to random initialization),
        # 'predicted_texts': [[b'build function token', b'build function test', b'build function is']]
        print(smart_compose_model.prefix_aware_beam_search({
            InputFtrType.TARGET_COLUMN_NAME: tf.constant(['build f'])
        }))


if __name__ == '__main__':
    tf.test.main()
