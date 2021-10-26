import sys

import tensorflow as tf

from smart_compose.run_smart_compose import main
from smart_compose.utils.parsing_utils import InputFtrType
from smart_compose.utils.testing.test_case import TestCase


class TestRun(TestCase):
    target_column_name = 'query'

    dataset_args = [
        "--test_file", TestCase.data_dir,
        "--dev_file", TestCase.data_dir,
        "--train_file", TestCase.data_dir,
        "--out_dir", TestCase.out_dir,

        "--num_train_steps", "300",
        "--steps_per_stats", "100",
        "--steps_per_eval", "100",

        "--train_batch_size", "10",
        "--test_batch_size", "10",

        "--resume_training", "False",
        "--distribution_strategy", "one_device",
        "--num_gpu", "0",
        "--run_eagerly", "True",
    ]

    optimization_args = [
        "--learning_rate", "0.01",
        "--optimizer", "adamw",
        "--pmetric", "perplexity"
    ]

    network_args = [
        "--num_units", "30",
    ]

    feature_args = [
        f"--{InputFtrType.TARGET_COLUMN_NAME}", target_column_name,
        "--vocab_file", TestCase.vocab_file,

        "--max_len", "16",
        "--min_len", "3",
    ]

    args = dataset_args + feature_args + network_args + optimization_args

    def testRunSmartCompose(self):
        sys.argv[1:] = self.args
        model = main(sys.argv)

        # {'exist_prefix': True,
        # 'predicted_scores':
        #       [[-9.4395971e+00, -1.0921703e+01, -1.1221247e+01, -1.1409840e+01,
        #         -1.3343969e+01, -1.7302521e+01, -1.0000000e+07, -1.0000000e+07,
        #         -1.0000000e+07, -1.0000000e+07]],
        # 'predicted_texts':
        #       [[b'this is a [SEP]', b'this is [UNK] [SEP]',
        #         b'this is sentence [SEP]', b'this is [SEP] [PAD]',
        #         b'this is function [SEP]', b'this is [PAD] [PAD]',
        #         b'[PAD] [PAD] [PAD] [PAD]', b'[PAD] [PAD] [PAD] [PAD]',
        #         b'[PAD] [PAD] [PAD] [PAD]', b'[PAD] [PAD] [PAD] [PAD]']],
        # }
        print(model({self.target_column_name: ["this is"]}))
        # {'exist_prefix': True,
        # 'predicted_scores':
        #       [[-3.75747681e-04, -9.05515480e+00, -1.13139982e+01,
        #         -1.16390209e+01, -1.19552555e+01, -1.38605614e+01,
        #         -1.38952370e+01, -1.43704872e+01, -1.00000000e+07,
        #         -1.00000000e+07]],
        # 'predicted_texts':
        #       [[b'[CLS] word function [SEP]', b'[CLS] word [SEP] [PAD]',
        #         b'[CLS] word [UNK] [SEP]', b'[CLS] word is [SEP]',
        #         b'[CLS] word sentence [SEP]', b'[CLS] word source [SEP]',
        #         b'[CLS] word build [SEP]', b'[CLS] word [PAD] [PAD]',
        #         b'[PAD] [PAD] [PAD] [PAD]', b'[PAD] [PAD] [PAD] [PAD]']],
        # }
        print(model({self.target_column_name: ["word"]}))
        # {'exist_prefix': True,
        # 'predicted_scores':
        #       [[-3.7574768e-04, -9.0551538e+00, -1.0219677e+01, -1.0755968e+01,
        #         -1.0904562e+01, -1.1283651e+01, -1.1313998e+01, -1.1424127e+01,
        #         -1.1639020e+01, -1.1955254e+01]],
        # 'predicted_texts':
        #       [[b'word function [SEP] [PAD]', b'word [SEP] [PAD] [PAD]',
        #         b'word function function [SEP]', b'word function sentence [SEP]',
        #         b'word is function [SEP]', b'word word function [SEP]',
        #         b'word [UNK] [SEP] [PAD]', b'word function [UNK] [SEP]',
        #         b'word is [SEP] [PAD]', b'word sentence [SEP] [PAD]']],
        # }
        print(model({self.target_column_name: ["word "]}))
        # {'exist_prefix': True,
        # 'predicted_scores':
        #       [[-1.3312817e-02, -6.3512154e+00, -6.8055477e+00, -7.4862165e+00,
        #         -1.3018910e+01, -1.3128208e+01, -1.3179526e+01, -1.3493284e+01,
        #         -1.3816385e+01, -1.3923851e+01]],
        # 'predicted_texts':
        #       [[b'word build [SEP] [PAD]', b'word build [UNK] [SEP]',
        #         b'word build function [SEP]', b'word build sentence [SEP]',
        #         b'word build a [SEP]', b'word build test [SEP]',
        #         b'word build word [SEP]', b'word build [PAD] [PAD]',
        #         b'word build build [SEP]', b'word build is [SEP]']]
        # }
        print(model({self.target_column_name: ["word b"]}))

        self._cleanUp(TestCase.out_dir)


if __name__ == '__main__':
    tf.test.main()
