import re

import tensorflow as tf
import tensorflow_addons.optimizers as tfa_optimizers
from official.nlp.optimization import WarmUp
from official.staging.training.grad_utils import _run_callbacks, _filter_and_allreduce_gradients

BERT_VAR_PREFIX = 'bert'


def create_optimizer(init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     optimizer,
                     use_lr_schedule,
                     use_bias_correction_for_adamw=False):
    """Creates an optimizer with learning rate schedule.

    Extended based on official.nlp.optimization.create_optimizer
    :param init_lr Initial learning rate
    :param num_train_steps Number of training steps
    :param num_warmup_steps Number of warmup steps
    :param optimizer Type of optimizer
    :param use_lr_schedule Whether to use learning rate scheudling such as warm up and decay
    :param use_bias_correction_for_adamw Whether to use bias correction in AdamWeightDecay optimzer
    """
    lr_schedule = init_lr
    if use_lr_schedule:
        # Implements linear decay of the learning rate.
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=init_lr,
            decay_steps=num_train_steps,
            power=1.0,
            end_learning_rate=0.0)

        if num_warmup_steps:
            lr_schedule = WarmUp(
                initial_learning_rate=init_lr,
                decay_schedule_fn=lr_schedule,
                warmup_steps=num_warmup_steps)

    optimizer_dct = {
        'sgd': tf.keras.optimizers.SGD(learning_rate=lr_schedule),
        'adam': tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        'adamw': AdamWeightDecay(learning_rate=lr_schedule,
                                 weight_decay_rate=0.01,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-6,
                                 exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'],
                                 use_bias_correction=use_bias_correction_for_adamw),
        'lamb': tfa_optimizers.LAMB(learning_rate=lr_schedule,
                                    weight_decay_rate=0.01,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-6,
                                    exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
    }

    return optimizer_dct[optimizer]


def process_grads_and_vars_without_explicit_allreduce(tape, optimizer, loss, trainable_variables, post_allreduce_callbacks):
    """Performs gradient allreduce relying on implicit allreduce in optimizer.apply_gradients()

    Reference: run_customized_training_loop() in official/nlp/bert/model_training_utils.py

    :param tape: An instance of `tf.GradientTape`.
    :param optimizer: An instance of `tf.keras.optimizers.Optimizer`.
    :param loss: the loss tensor.
    :param trainable_variables: A list of model Variables.
    :param post_allreduce_callbacks: A list of callback functions that takes gradients and model variables pairs as input, manipulate them, and
      returns a new gradients and model variables paris. The callback functions will be invoked in the list order and right before gradients
      are applied to variables for updates. Default is no callbacks.
    """
    if isinstance(optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer):
        # FP16 GPU code path
        with tape:
            scaled_loss = optimizer.get_scaled_loss(loss)
        scaled_grads = tape.gradient(scaled_loss, trainable_variables)
        grads = optimizer.get_unscaled_gradients(scaled_grads)
    else:
        # FP32 GPU code path
        grads = tape.gradient(loss, trainable_variables)

    trainable_vars = trainable_variables
    grads_and_vars = zip(grads, trainable_vars)
    if post_allreduce_callbacks:
        grads_and_vars = _run_callbacks(post_allreduce_callbacks, zip(grads, trainable_vars))
    return grads_and_vars


def process_grads_and_vars_using_explicit_allreduce(tape,
                                                    optimizer,
                                                    loss, trainable_variables,
                                                    pre_allreduce_callbacks=None,
                                                    post_allreduce_callbacks=None):
    """Explicitly performs gradient allreduce, instead of relying on implicit allreduce in optimizer.apply_gradients()

    If training using FP16 mixed precision, explicit allreduce will aggregate gradients in FP16 format.
    For TPU and GPU training using FP32, explicit allreduce will aggregate gradients in FP32 format.

    Reference: minimize_using_explicit_allreduce() in official/staging/training/grad_utils.py

    :param tape: An instance of `tf.GradientTape`.
    :param optimizer: An instance of `tf.keras.optimizers.Optimizer`.
    :param loss: the loss tensor.
    :param trainable_variables: A list of model Variables.
    :param pre_allreduce_callbacks: A list of callback functions that takes gradients and model variables pairs as input, manipulate them, and returns a new
      gradients and model variables pairs. The callback functions will be invoked in the list order and before gradients are allreduced.
      With mixed precision training, the pre_allreduce_allbacks will be applied on scaled_gradients. Default is no callbacks.
    :param post_allreduce_callbacks: A list of callback functions that takes gradients and model variables pairs as input, manipulate them, and
      returns a new gradients and model variables paris. The callback functions will be invoked in the list order and right before gradients
      are applied to variables for updates. Default is no callbacks.
    """
    if isinstance(optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer):
        # FP16 GPU code path
        with tape:
            scaled_loss = optimizer.get_scaled_loss(loss)

        scaled_grads = tape.gradient(scaled_loss, trainable_variables)
        grads_and_vars = zip(scaled_grads, trainable_variables)
        if pre_allreduce_callbacks:
            grads_and_vars = _run_callbacks(pre_allreduce_callbacks, grads_and_vars)
        (allreduced_scaled_grads, filtered_trainable_variables) = _filter_and_allreduce_gradients(grads_and_vars, allreduce_precision="float16")
        allreduced_unscaled_grads = optimizer.get_unscaled_gradients(allreduced_scaled_grads)
        grads_and_vars = zip(allreduced_unscaled_grads, filtered_trainable_variables)
    else:
        # TPU or FP32 GPU code path
        grads = tape.gradient(loss, trainable_variables)
        grads_and_vars = zip(grads, trainable_variables)
        if pre_allreduce_callbacks:
            grads_and_vars = _run_callbacks(pre_allreduce_callbacks, grads_and_vars)
        (allreduced_grads, filtered_trainable_variables) = _filter_and_allreduce_gradients(grads_and_vars, allreduce_precision="float32")
        grads_and_vars = zip(allreduced_grads, filtered_trainable_variables)
    if post_allreduce_callbacks:
        grads_and_vars = _run_callbacks(post_allreduce_callbacks, grads_and_vars)

    return grads_and_vars


def split_bert_grads_and_vars(grads_and_vars):
    """ Splits gradient and variables into bert related and non related groups"""
    non_bert_grads_and_vars = list()
    bert_grads_and_vars = list()
    for grad, var in grads_and_vars:
        if BERT_VAR_PREFIX in var.name:
            bert_grads_and_vars.append((grad, var))
        else:
            non_bert_grads_and_vars.append((grad, var))
    return bert_grads_and_vars, non_bert_grads_and_vars


def clip_by_global_norm(grads_and_vars, clip_norm):
    """Returns the gradient clipping function that clips gradients with given clipnorm"""
    grads, variables = zip(*grads_and_vars)
    (clipped_grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    return zip(clipped_grads, variables)


class AdamWeightDecay(tf.keras.optimizers.Adam):
    """Adam enables L2 weight decay and clip_by_global_norm on gradients.

    Just adding the square of the weights to the loss function is *not* the
    correct way of using L2 regularization/weight decay with Adam, since that will
    interact with the m and v parameters in strange ways.

    Instead we want ot decay the weights in a manner that doesn't interact with
    the m/v parameters. This is equivalent to adding the square of the weights to
    the loss with plain (non-momentum) SGD.

    This class is extended based on the implementation of tf-models-offical. This class adds a use_bias_correction param so that users
        can disable the bias to speed up training. It's worth noting that the TF1 version of AdamWeightDecay also does not have bias
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 weight_decay_rate=0.0,
                 include_in_weight_decay=None,
                 exclude_from_weight_decay=None,
                 name='AdamWeightDecay',
                 use_bias_correction=False,
                 **kwargs):
        super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2,
                                              epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay
        self.use_bias_correction = use_bias_correction

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its config with WarmUp custom object."""
        custom_objects = {'WarmUp': WarmUp}
        return super(AdamWeightDecay, cls).from_config(
            config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
                                                    apply_state)
        apply_state[(var_device, var_dtype)]['weight_decay_rate'] = tf.constant(
            self.weight_decay_rate, name='adam_weight_decay_rate')

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var *
                apply_state[(var.device, var.dtype.base_dtype)]['weight_decay_rate'],
                use_locking=self._use_locking)
        return tf.no_op()

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        grads, tvars = list(zip(*grads_and_vars))
        if experimental_aggregate_gradients:
            # when experimental_aggregate_gradients = False, apply_gradients() no
            # longer implicitly allreduce gradients, users manually allreduce gradient
            # and passed the allreduced grads_and_vars. For now, the
            # clip_by_global_norm will be moved to before the explicit allreduce to
            # keep the math the same as TF 1 and pre TF 2.2 implementation.
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        return super(AdamWeightDecay, self).apply_gradients(
            zip(grads, tvars),
            name=name,
            experimental_aggregate_gradients=experimental_aggregate_gradients)

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients
        if not self.use_bias_correction:
            # Explicitly setting the beta powers to be 0 so that we do not do bias correction for the estimations
            # Below are the standard update for Adam
            # $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
            # $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
            # $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
            # $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
            #
            # By setting beta_1_power and beta_2_power to 0, the update is equivalent to the TF1 implementation of Adam
            coefficients['beta_1_power'] = .0
            coefficients['beta_2_power'] = .0
        return coefficients['lr_t'], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay,
                         self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay,
                         self)._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        config = super(AdamWeightDecay, self).get_config()
        config.update({
            'weight_decay_rate': self.weight_decay_rate,
        })
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True
