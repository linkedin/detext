import re
import tensorflow as tf
from os.path import join as path_join
from detext.utils import executor_utils


def create_optimizer(hparams, loss):
    """
    Creates an optimizer training op.
    If the parameter lr_bert is specified, then use another adam for this learning rate.
    """
    tvars = tf.trainable_variables()

    if hparams.use_horovod:
        import horovod.tensorflow as hvd

    # Log trainable variables (with local mode, or chief for ps strategy or rank 0 for hvd training)
    task_type = executor_utils.get_executor_task_type()
    if (hparams.use_horovod is False and task_type in [executor_utils.CHIEF, executor_utils.LOCAL_MODE]) or \
            (hparams.use_horovod is True and hvd.rank() == 0):
        with tf.gfile.Open(path_join(hparams.out_dir, 'network_structure.txt'), 'w') as fout:
            fout.write("# Trainable variables\n")
            total_deep_params = 0
            total_params = 0
            for param in tvars:
                psize = 1
                for s in param.get_shape():
                    psize *= s
                total_params += psize
                if param.name.startswith(hparams.ftr_ext):
                    total_deep_params += psize
                fout.write("  %s, %s, %s\n" % (param.name, str(param.get_shape()), param.op.device))
            fout.write('\n')
            fout.write('# Total trainable params: {}\n'.format(total_params))
            fout.write('# Out of the total trainable params, the total {} parameters: {}\n'
                       .format(hparams.ftr_ext, total_deep_params))

    # Define optimizer parameters
    init_lr = hparams.learning_rate
    num_train_steps = hparams.num_train_steps
    num_warmup_steps = hparams.num_warmup_steps
    lr_bert = hparams.lr_bert

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    if hparams.optimizer.startswith("bert_"):
        # Using optimizer with bert's implementation
        # Implements linear decay of the learning rate.
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)

        # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
        # learning rate will be `global_step/num_warmup_steps * init_lr`.
        if num_warmup_steps:
            global_steps_int = tf.cast(global_step, tf.int32)
            warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = init_lr * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

        name_2_optimizer = {
            'bert_adam': AdamWeightDecayOptimizer,
            'bert_lamb': LAMBOptimizer
        }
        OptimizerFunc = name_2_optimizer[hparams.optimizer]

        # It is recommended that you use this optimizer for fine tuning, since this
        # is how the model was trained (note that the Adam/Lamb m/v variables are NOT
        # loaded from init_checkpoint.)
        optimizer = OptimizerFunc(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        if hparams.use_horovod:
            # Horovod's distributed optimizer handles allreduce calls, synchronous only
            optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=True)
            grads_and_vars = optimizer.compute_gradients(loss, tvars)
            grads = [grad for grad, var in grads_and_vars]
            tvars = [var for grad, var in grads_and_vars]
        else:
            grads = tf.gradients(loss, tvars)

        grads, grad_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)

        if lr_bert is None:
            # If not a separate learning rate for bert (lr_bert) is specified,
            # all components use the same learning rate
            train_op = optimizer.apply_gradients(
                zip(grads, tvars), global_step=global_step)

            # Normally the global step update is done inside of `apply_gradients`.
            # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
            # a different optimizer, you should probably take this line out.
            new_global_step = global_step + 1
            train_op = tf.group(train_op, [global_step.assign(new_global_step)])
        else:
            # the BERT components will use another learning rate
            optimizer_bert = OptimizerFunc(
                learning_rate=learning_rate * lr_bert / init_lr,
                weight_decay_rate=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
            if hparams.use_horovod:
                # Treat the bert optimizer the same as the original optimizer: wrapped with horovod
                optimizer_bert = hvd.DistributedOptimizer(optimizer_bert, sparse_as_dense=True)

            bert_grad, bert_tvars = [], []
            other_grad, other_tvars = [], []
            for grad, tvar in zip(grads, tvars):
                if tvar is not None and grad is not None:
                    if tvar.name.startswith('bert'):
                        bert_grad.append(grad)
                        bert_tvars.append(tvar)
                        print('****bert param:', tvar.name)
                    else:
                        other_grad.append(grad)
                        other_tvars.append(tvar)
                        print('****other param:', tvar.name)
            print('--------------\n', '# of bert', len(bert_grad), '# of other', len(other_grad), '\n--------------')
            bert_train_op = optimizer_bert.apply_gradients(
                zip(bert_grad, bert_tvars), global_step=global_step)
            other_train_op = optimizer.apply_gradients(
                zip(other_grad, other_tvars), global_step=global_step)

            new_global_step = global_step + 1
            train_op = tf.group(bert_train_op, other_train_op, [global_step.assign(new_global_step)])

        return train_op, grad_norm, learning_rate

    elif hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    elif hparams.optimizer == "adam":
        opt = tf.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError("Only support sgd/adam/bert_adam as optimizer option")

    # Gradients
    gradients = tf.gradients(loss, tvars, colocate_gradients_with_ops=True)
    clipped_gradients, grad_norm = tf.clip_by_global_norm(gradients, hparams.max_gradient_norm)
    train_op = opt.apply_gradients(zip(clipped_gradients, tvars), global_step=global_step)

    return train_op, grad_norm, learning_rate


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


class LAMBOptimizer(tf.train.Optimizer):
    """
    Optimizer that implements the Layer-wise Adaptive Moments (LAMB).
    See paper [Large Batch Optimization for Deep Learning: Training BERT
        in 76 minutes](https://arxiv.org/abs/1904.00962).
    Official implementation from tensorflow addon tensorflow_addons/optimizers/lamb.py

    Note that we does not apply adam bias correction for the moments estimates to keep it similar to the original AdamW
    optimizer in BERT pretraining (which is the AdamWeightDecayOptimizer here). The only difference for the
    LAMBOptimizer compared to AdamWeightDecayOptimizer implementation is the added layer adaptation.
    """

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="LAMBOptimizer"):
        """Constructs a LAMBOptimizer."""
        super(LAMBOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/lamb_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/lamb_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            # lamb layer adaptation
            w_norm = tf.norm(param, ord=2)
            g_norm = tf.norm(update, ord=2)

            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

            update_with_lr = self.learning_rate * ratio * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
