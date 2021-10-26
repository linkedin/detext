import tensorflow as tf

from detext.layers import representation_layer
from detext.utils.parsing_utils import InputFtrType, InternalFtrType
from detext.utils.testing.data_setup import DataSetup


class TestRepresentationLayer(tf.test.TestCase, DataSetup):
    """Unit test for representation_layer.py """
    min_len = 3
    max_len = 4

    ftr_ext = 'cnn'
    text_encoder_param = DataSetup.cnn_param

    def testRepresentationLayer(self):
        """Tests RepLayer"""
        doc_id_fields_lst = [self.ranking_doc_id_fields, None]
        num_doc_id_fields_lst = [self.num_doc_id_fields, 0]
        user_id_fields_lst = [self.user_id_fields, None]
        num_user_id_fields_lst = [self.num_user_id_fields, 0]

        for doc_id_fields, user_id_fields, num_doc_id_fields, num_user_id_fields in zip(doc_id_fields_lst, user_id_fields_lst, num_doc_id_fields_lst,
                                                                                        num_user_id_fields_lst):
            add_doc_projection = False
            add_user_projection = False
            self._testRepresentationLayer(add_doc_projection, add_user_projection, doc_id_fields, user_id_fields, num_doc_id_fields, num_user_id_fields)

            add_doc_projection = True
            add_user_projection = False
            self._testRepresentationLayer(add_doc_projection, add_user_projection, doc_id_fields, user_id_fields, num_doc_id_fields, num_user_id_fields)

            add_doc_projection = True
            add_user_projection = True
            self._testRepresentationLayer(add_doc_projection, add_user_projection, doc_id_fields, user_id_fields, num_doc_id_fields, num_user_id_fields)

    def _testRepresentationLayer(self, add_doc_projection, add_user_projection, doc_id_fields, user_id_fields, num_doc_id_fields, num_user_id_fields):
        """Tests RepLayer given input"""
        layer = representation_layer.RepresentationLayer(self.ftr_ext, self.num_doc_fields, self.num_user_fields,
                                                         num_doc_id_fields, num_user_id_fields, add_doc_projection, add_user_projection,
                                                         self.text_encoder_param, self.id_encoder_param)
        outputs = layer(
            {InputFtrType.QUERY_COLUMN_NAME: self.query, InputFtrType.DOC_TEXT_COLUMN_NAMES: self.ranking_doc_fields,
             InputFtrType.USER_TEXT_COLUMN_NAMES: self.user_fields,
             InputFtrType.DOC_ID_COLUMN_NAMES: doc_id_fields, InputFtrType.USER_ID_COLUMN_NAMES: user_id_fields}, False)

        self.assertEqual(tf.shape(outputs[InternalFtrType.QUERY_FTRS])[-1], layer.ftr_size)
        self.assertAllEqual(tf.shape(outputs[InternalFtrType.DOC_FTRS])[-2:], [layer.output_num_doc_fields, layer.ftr_size])
        self.assertAllEqual(tf.shape(outputs[InternalFtrType.USER_FTRS])[-2:], [layer.output_num_user_fields, layer.ftr_size])


if __name__ == '__main__':
    tf.test.main()
