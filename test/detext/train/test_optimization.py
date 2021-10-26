import tensorflow as tf

from detext.train import optimization
from detext.train.optimization import BERT_VAR_PREFIX


class TestOptimization(tf.test.TestCase):
    @classmethod
    def _get_loss(cls, x, y_true, model, loss_obj):
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = loss_obj(y_true, y_pred)
        return loss, tape

    @classmethod
    def _minimize(cls, x, y_true, model, loss_obj, optimizer):
        loss, tape = cls._get_loss(x, y_true, model, loss_obj)

        train_vars = model.trainable_variables
        grads = tape.gradient(loss, train_vars)
        grads_and_vars = zip(grads, train_vars)
        optimizer.apply_gradients(grads_and_vars)
        return loss

    def _train_linear_model(self, x, y_true, init_lr, num_train_steps, num_warmup_steps, process_grads_and_vars_fn):
        """Helper function to train a linear model"""
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer='sgd',
                                                  use_lr_schedule=True,
                                                  use_bias_correction_for_adamw=False)

        model = tf.keras.Sequential(tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.zeros()))
        loss_obj = tf.keras.losses.MeanSquaredError()

        for _ in range(2):
            loss, tape = self._get_loss(x, y_true, model, loss_obj)
            grads_and_vars = process_grads_and_vars_fn(tape, optimizer, loss, model.trainable_variables, [])
            optimizer.apply_gradients(grads_and_vars)
        return model

    def testProcessGradsAndVars(self):
        """Tests process_grads_and_vars with/without explicit allreduce"""
        init_lr = 0.05
        num_train_steps = 10
        num_warmup_steps = 3

        x = tf.constant([[0.1, 0.2], [0.3, 0.1]], dtype=tf.float32)
        y_true = x[:, 0] + x[:, 1]

        model_explicit_allreduce = self._train_linear_model(x, y_true, init_lr, num_train_steps, num_warmup_steps,
                                                            optimization.process_grads_and_vars_using_explicit_allreduce)
        model_implicit_allreduce = self._train_linear_model(x, y_true, init_lr, num_train_steps, num_warmup_steps,
                                                            optimization.process_grads_and_vars_without_explicit_allreduce)

        self.assertAllEqual([x.numpy() for x in model_explicit_allreduce.trainable_variables],
                            [x.numpy() for x in model_implicit_allreduce.trainable_variables])

    def testSplitBertGradsAndVars(self):
        """ Tests split_bert_grads_and_vars() """
        grads = [tf.constant(3.0, dtype='float32'), tf.constant(4.0, dtype='float32')]
        variables = [tf.Variable(1.0, name='rep_model/' + BERT_VAR_PREFIX + '_model/some_var'), tf.Variable(1.0, name='some_var')]
        bert_grads_and_vars, non_bert_grads_and_vars = optimization.split_bert_grads_and_vars(zip(grads, variables))

        self.assertAllEqual(bert_grads_and_vars, [(grads[0], variables[0])])
        self.assertAllEqual(non_bert_grads_and_vars, [(grads[1], variables[1])])

    def testClipByGlobalNorm(self):
        """ Tests clip_by_global_norm() """
        grads = [tf.constant(3.0, dtype='float32'), tf.constant(4.0, dtype='float32')]
        variables = [tf.Variable(1.0), tf.Variable(1.0)]

        clip_norm_lst = [1.0, 50.0]
        expected_grads_lst = [[g / 5 for g in grads], [g for g in grads]]
        assert len(clip_norm_lst) == len(expected_grads_lst)

        for clip_norm, expected_grads in zip(clip_norm_lst, expected_grads_lst):
            self._testClipByGlobalNorm(grads, variables, clip_norm, expected_grads)

    def _testClipByGlobalNorm(self, grads, variables, clip_norm, expected_grads):
        """ Tests clip_by_global_norm() given clip norm"""
        grads_and_vars = optimization.clip_by_global_norm(zip(grads, variables), clip_norm)
        result_grad = [x[0] for x in grads_and_vars]
        self.assertAllEqual(result_grad, expected_grads)

    def testCreateOptimizer(self):
        """ Tests create_optimizer() """
        init_lr = 0.05
        num_train_steps = 10
        num_warmup_steps = 3
        num_bp_steps = 5

        x = tf.constant([[0.1, 0.2], [0.3, 0.1]], dtype=tf.float32)
        y_true = x[:, 0] + x[:, 1]

        for optimizer_type in ['sgd', 'adam', 'adamw', 'lamb']:
            optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                      num_train_steps=num_train_steps,
                                                      num_warmup_steps=num_warmup_steps,
                                                      optimizer=optimizer_type,
                                                      use_lr_schedule=True,
                                                      use_bias_correction_for_adamw=False)

            model = tf.keras.Sequential(tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.zeros()))
            loss_obj = tf.keras.losses.MeanSquaredError()

            prev_loss = self._minimize(x, y_true, model, loss_obj, optimizer).numpy()
            prev_lr = optimizer._decayed_lr('float32').numpy()
            for step in range(1, num_bp_steps):
                loss = self._minimize(x, y_true, model, loss_obj, optimizer).numpy()

                # When warm up steps > 0, lr will be 0 when calculating prev_loss and therefore no backprop will be executed
                # This will cause loss_at_step_0 = prev_loss
                if step > 1:
                    self.assertLess(loss, prev_loss, f"Loss should be declining at each step. Step:{step}")

                # Learning rate check
                lr = optimizer._decayed_lr('float32').numpy()
                if step < num_warmup_steps:
                    self.assertGreater(lr, prev_lr, f"Learning rate should be increasing during warm up. Step:{step}")
                else:
                    self.assertLess(lr, prev_lr, f"Learning rate should be decreasing after warm up. Step:{step}")

                prev_loss = loss
                prev_lr = lr


if __name__ == '__main__':
    tf.test.main()
