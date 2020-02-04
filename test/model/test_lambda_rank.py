import tensorflow as tf
from detext.model import lambdarank
from detext.utils import test_utils


class TestLambdaRank(tf.test.TestCase):
    """Unit test for lambdarank."""

    def testPairwiseRank(self):
        """test pairwise ranking without lambda in compute_ndcg()"""
        pairwise_rank = lambdarank.LambdaRank(None)
        scores = tf.placeholder(dtype=tf.float32, shape=[None, None])
        labels = tf.placeholder(dtype=tf.float32, shape=[None, None])
        group_size = tf.placeholder(dtype=tf.int32, shape=[None])
        loss, _ = pairwise_rank(scores, labels, group_size)

        with self.test_session() as sess:
            # a simple case
            loss_v = sess.run(loss, feed_dict={
                scores: [[0, 1]],
                labels: [[0, 1]],
                group_size: [2],
            })
            self.assertAllClose([[[0, 0], [test_utils.neg_log_sigmoid(1), 0]]], loss_v)

            # a more complicated case
            loss_v = sess.run(loss, feed_dict={
                scores: [[2, 1, 3, 0],
                         [5, 2, 7, 9]],
                labels: [[1, 0, 0, 2],
                         [1, 3, 2, 0]],
                group_size: [2, 3],
            })
            self.assertAllClose([[[0, test_utils.neg_log_sigmoid(1), 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]],
                                 [[0, 0, 0, 0],
                                  [test_utils.neg_log_sigmoid(-3), 0, test_utils.neg_log_sigmoid(-5), 0],
                                  [test_utils.neg_log_sigmoid(2), 0, 0, 0],
                                  [0, 0, 0, 0]]], loss_v)
            # another case
            loss_v = sess.run(loss, feed_dict={
                scores: [[4, 8, 7, 9, 5, 4.5, 3, 1, 10, 2]],
                labels: [[3, 2, 3, 0, 0, 1, 2, 2, 3, 0]],
                group_size: [10],
            })
            loss2 = test_utils.get_lambda_loss(scores=[[4, 8, 7, 9, 5, 4.5, 3, 1, 10, 2]],
                                               labels=[[3, 2, 3, 0, 0, 1, 2, 2, 3, 0]],
                                               group_size=[10])
            self.assertAllClose(loss2, loss_v)

    def testLambdaRank(self):
        """test pairwise ranking with lambda rank"""
        lambda_rank = lambdarank.LambdaRank({'metric': 'ndcg', 'topk': 4})
        scores = tf.placeholder(dtype=tf.float32, shape=[None, None])
        labels = tf.placeholder(dtype=tf.float32, shape=[None, None])
        group_size = tf.placeholder(dtype=tf.int32, shape=[None])
        loss, _ = lambda_rank(scores, labels, group_size)
        lambda_rank_top10 = lambdarank.LambdaRank({'metric': 'ndcg', 'topk': 10})
        loss_top10, _ = lambda_rank_top10(scores, labels, group_size)

        with self.test_session() as sess:
            # a simple case
            loss_v = sess.run(loss, feed_dict={
                scores: [[0, 1]],
                labels: [[0, 1]],
                group_size: [2],
            })
            loss2 = test_utils.get_lambda_loss(scores=[[0, 1]], labels=[[0, 1]], group_size=[2], topk=4)
            self.assertAllClose(loss2, loss_v)

            # a more complicated case
            loss_v = sess.run(loss, feed_dict={
                scores: [[4, 1, 3, 0],
                         [5, 2, 7, 9]],
                labels: [[1, 0, 0, 2],
                         [1, 3, 2, 2]],
                group_size: [4, 3],
            })
            loss2 = test_utils.get_lambda_loss(scores=[[4, 1, 3, 0],
                                                       [5, 2, 7, 9]],
                                               labels=[[1, 0, 0, 2],
                                                       [1, 3, 2, 2]],
                                               group_size=[4, 3],
                                               topk=4)
            self.assertAllClose(loss2, loss_v)

            # another case
            loss_v = sess.run(loss_top10, feed_dict={
                scores: [[4, 8, 7, 9, 5, 4.5, 3, 1, 10, 2]],
                labels: [[3, 2, 3, 0, 0, 1, 2, 2, 3, 0]],
                group_size: [10],
            })
            loss2 = test_utils.get_lambda_loss(scores=[[4, 8, 7, 9, 5, 4.5, 3, 1, 10, 2]],
                                               labels=[[3, 2, 3, 0, 0, 1, 2, 2, 3, 0]],
                                               group_size=[10],
                                               topk=10)
            self.assertAllClose(loss2, loss_v)


if __name__ == "__main__":
    tf.test.main()
