import tensorflow as tf
from detext.model.softmax_loss import compute_softmax_loss
from detext.utils import test_utils


class TestSoftmaxLoss(tf.test.TestCase):
    """Unit test for lambdarank."""

    def testCompute(self):
        """test pairwise ranking without lambda in compute_ndcg()"""
        scores = tf.placeholder(dtype=tf.float32, shape=[None, None])
        labels = tf.placeholder(dtype=tf.float32, shape=[None, None])
        group_size = tf.placeholder(dtype=tf.int32, shape=[None])
        loss = compute_softmax_loss(scores, labels, group_size)

        with self.test_session() as sess:
            # a simple case
            loss_v = sess.run(loss, feed_dict={
                scores: [[0, 1]],
                labels: [[0, 1]],
                group_size: [2],
            })
            self.assertAllClose(test_utils.get_softmax_loss([[0, 1]], [[0, 1]], [2]), loss_v)

            # a more complicated case
            loss_v = sess.run(loss, feed_dict={
                scores: [[2, 1, 3, 0],
                         [5, 2, 7, 9]],
                labels: [[1, 0, 0, 2],
                         [1, 3, 2, 0]],
                group_size: [2, 3],
            })
            loss_v2 = test_utils.get_softmax_loss(
                [[2, 1, 3, 0],
                 [5, 2, 7, 9]],
                [[1, 0, 0, 2],
                 [1, 3, 2, 0]],
                [2, 3]
            )
            self.assertAllClose(loss_v2, loss_v)


if __name__ == "__main__":
    tf.test.main()
