import tensorflow as tf

from smart_compose.layers import prefix_search
from smart_compose.layers import vocab_layer
from smart_compose.utils.parsing_utils import InternalFtrType
from smart_compose.utils.testing.test_case import TestCase


class TestPrefixSearch(TestCase):
    """ Unit test for prefix_search.py """
    min_len = 0
    max_len = 3
    num_cls = 1
    num_sep = 0
    searcher = prefix_search.PrefixSearcher(vocab_layer.create_vocab_layer(TestCase.vocab_layer_param, ''),
                                            min_len, max_len, num_cls, num_sep)

    def testPrefixSearch(self):
        prefix_list = [
            tf.constant('b'), tf.constant('build s'), tf.constant('h'), tf.constant(''), tf.constant('b '), tf.constant(' ')
        ]
        exist_prefix_list = [
            True, True, False, False, True, True
        ]
        vocab_mask_list = [
            [False] * 4 + [True] + [False] * 11,
            [False] * 12 + [True, True] + [False] * 2,
            [False] * 16,
            [False] * 16,
            [True] * 16,
            [True] * 16
        ]
        length_list = [1, 2, 1, 0, 2, 1]

        assert len(prefix_list) == len(exist_prefix_list) == len(vocab_mask_list) == len(length_list), 'Test input list must have the same size'
        for prefix, exist_prefix, vocab_mask, length in zip(prefix_list, exist_prefix_list, vocab_mask_list, length_list):
            self._testPrefixSearch(prefix, exist_prefix, vocab_mask, length)

    def _testPrefixSearch(self, prefix, exist_prefix, vocab_mask, length):
        outputs = self.searcher(prefix)

        self.assertAllEqual(outputs[InternalFtrType.EXIST_PREFIX], exist_prefix)
        self.assertAllEqual(outputs[InternalFtrType.COMPLETION_VOCAB_MASK], vocab_mask)
        self.assertAllEqual(outputs[InternalFtrType.LENGTH], length)

    def testKeyValueArrayDict(self):
        keys_list = [[1, 2, 3],
                     [1, 2, 3],
                     ['10', '2', '3']]
        values_list = [[[2, 3, 4], [5, 0], [1]],
                       [[2, 3, 4], [5, 0], [1]],
                       [['2', '3', '4'], ['5', '0'], ['1']]]
        test_key_list = [2, -1, '2']
        default_values = [-1, -1, ""]
        expected_value_list = [
            tf.convert_to_tensor(
                [0, default_values[0], default_values[0], default_values[0], default_values[0], 5],
                dtype='int32'
            ),
            tf.convert_to_tensor(
                [0, default_values[1], default_values[1], default_values[1], default_values[1], 5],
                dtype='int32'
            ),
            tf.convert_to_tensor(
                ['0', default_values[2], default_values[2], default_values[2], default_values[2], '5'],
                dtype='string'
            )
        ]
        key_type_list = ['int32', 'int32', 'string']
        exist_prefix_list = [tf.convert_to_tensor(True), tf.convert_to_tensor(False), tf.convert_to_tensor(True)]
        for keys, values, test_key, expected_value, default_value, key_type, exist_prefix in zip(
                keys_list, values_list, test_key_list, expected_value_list, default_values, key_type_list, exist_prefix_list):
            self._testKeyValueArrayDict(keys, values, test_key, expected_value, default_value, key_type, exist_prefix)

    def _testKeyValueArrayDict(self, keys, values, test_key, expected_value, default_value, key_type, exist_prefix):
        table = prefix_search.KeyValueArrayDict(keys, values)
        outputs = table.lookup(tf.convert_to_tensor(test_key, dtype=key_type))
        self.assertAllEqual(outputs[InternalFtrType.EXIST_KEY], exist_prefix)
        if exist_prefix.numpy():
            self.assertAllEqual(tf.sparse.to_dense(outputs[InternalFtrType.COMPLETION_INDICES], default_value=default_value), expected_value)


if __name__ == '__main__':
    tf.test.main()
