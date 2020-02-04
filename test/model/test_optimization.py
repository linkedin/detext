import tensorflow as tf
from detext.train import optimization


class OptimizationTest(tf.test.TestCase):

    def test_adam(self):
        with self.test_session() as sess:
            w = tf.get_variable(
                "w",
                shape=[3],
                initializer=tf.constant_initializer([0.1, -0.2, -0.1]))
            x = tf.constant([0.4, 0.2, -0.5])
            loss = tf.reduce_mean(tf.square(x - w))
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)
            global_step = tf.train.get_or_create_global_step()
            optimizer = optimization.AdamWeightDecayOptimizer(learning_rate=0.2)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            for _ in range(100):
                sess.run(train_op)
            w_np = sess.run(w)
            self.assertAllClose(w_np.flat, [0.4, 0.2, -0.5], rtol=1e-2, atol=1e-2)

    def test_different_lr(self):
        """Test the usage of different learning rate."""
        with self.test_session() as sess:
            bert_w = tf.get_variable(
                "bert/w",
                shape=[3],
                initializer=tf.constant_initializer([0.1, 0.1, 0.1]))
            non_bert_w = tf.get_variable(
                "w",
                shape=[3],
                initializer=tf.constant_initializer([0.1, 0.1, 0.1]))
            x = tf.constant([1.0, 2.0, 3.0])
            loss = tf.reduce_mean(tf.square(x - bert_w - non_bert_w))

            hparams = tf.contrib.training.HParams(
                learning_rate=0.001,
                num_train_steps=100,
                num_warmup_steps=0,
                lr_bert=0.00001,
                optimizer="bert_adam"
            )
            train_op, _, _ = optimization.create_optimizer(hparams, loss)

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            sess.run(train_op)
            bert_w_v, non_bert_w_v = sess.run((bert_w, non_bert_w))
            print(bert_w_v, non_bert_w_v)
            # The difference of weight values (gradient) reflects the learning arte difference
            self.assertAllClose((bert_w_v - 0.1) / (non_bert_w_v - 0.1), [0.01, 0.01, 0.01], rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tf.test.main()
