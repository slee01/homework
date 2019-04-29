import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions
from tensorflow.python import keras
from tensorflow.python.keras.engine.network import Network


class QFunction(Network):
    def __init__(self, hidden_layer_sizes, **kwargs):
        super(QFunction, self).__init__(**kwargs)
        self._hidden_layer_sizes = hidden_layer_sizes

    def build(self, input_shape):
        inputs = [
            layers.Input(batch_shape=input_shape[0], name='observations'),
            layers.Input(batch_shape=input_shape[1], name='actions')
        ]

        x = layers.Concatenate(axis=1)(inputs)
        for hidden_units in self._hidden_layer_sizes:
            x = layers.Dense(hidden_units, activation='relu')(x)
        q_values = layers.Dense(1, activation=None)(x)

        self._init_graph_network(inputs, q_values)
        super(QFunction, self).build(input_shape)


class ValueFunction(Network):
    def __init__(self, hidden_layer_sizes, **kwargs):
        super(ValueFunction, self).__init__(**kwargs)
        self._hidden_layer_sizes = hidden_layer_sizes

    def build(self, input_shape):
        inputs = layers.Input(batch_shape=input_shape, name='observations')

        x = inputs
        for hidden_units in self._hidden_layer_sizes:
            x = layers.Dense(hidden_units, activation='relu')(x)
        values = layers.Dense(1, activation=None)(x)

        self._init_graph_network(inputs, values)
        super(ValueFunction, self).build(input_shape)


class GaussianPolicy(Network):
    def __init__(self, action_dim, hidden_layer_sizes, reparameterize, **kwargs):
        super(GaussianPolicy, self).__init__(**kwargs)
        self._action_dim = action_dim
        self._f = None
        self._hidden_layer_sizes = hidden_layer_sizes
        self._reparameterize = reparameterize

    def build(self, input_shape):
        inputs = layers.Input(batch_shape=input_shape, name='observations')

        x = inputs
        for hidden_units in self._hidden_layer_sizes:
            x = layers.Dense(hidden_units, activation='relu')(x)

        mean_and_log_std = layers.Dense(
            self._action_dim * 2, activation=None)(x)

        def create_distribution_layer(mean_and_log_std):
            mean, log_std = tf.split(
                mean_and_log_std, num_or_size_splits=2, axis=1)
            log_std = tf.clip_by_value(log_std, -20., 2.)

            distribution = distributions.MultivariateNormalDiag(
                loc=mean,
                scale_diag=tf.exp(log_std))

            raw_actions = distribution.sample()
            if not self._reparameterize:
                ### Problem 1.3.A
                ### YOUR CODE HERE
                raw_actions = tf.stop_gradient(raw_actions)

            log_probs = distribution.log_prob(raw_actions)
            log_probs -= self._squash_correction(raw_actions)

            ### Problem 2.A
            ### YOUR CODE HERE
            actions = tf.tanh(raw_actions)

            return actions, log_probs

        samples, log_probs = layers.Lambda(create_distribution_layer)(mean_and_log_std)
        # layers.lambda ==> wraps arbitrary expression as a Layer object

        self._init_graph_network(inputs=inputs, outputs=[samples, log_probs])
        super(GaussianPolicy, self).build(input_shape)

    def _squash_correction(self, raw_actions, stable=True, eps=1e-8):
        # The formula log(1 - tanh(x)^2) is mathematically equivalent to
        # 2 * (tf.log(2.0) - x - tf.nn.softplus(-2. * x)), which is numerically more stable
        # Derivation:
        #   log(1 - tanh(x)^2)
        # = log(sech(x)^2)
        # = 2 * log(sech(x))
        # = 2 * log(2e^-x / (e^-2x + 1))
        # = 2 * (log(2) - x - log(e^-2x + 1))
        # (1)  = 2 * (log(2) - x - softplus(-2x))

        # (2)  = 2 * (log(2) - x - (log(1 + e^2x) - log(e^2x)))
        # (2)  = 2 * (log(2) + x - softplus(+2x))

        ### Problem 2.B
        ### YOUR CODE HERE
        if not stable:
            correction = tf.reduce_sum(tf.log(1.0 - tf.square(tf.tanh(raw_actions)) + eps))
        else:
            correction = tf.log(4.) + 2. * (raw_actions - tf.nn.softplus(2. * raw_actions))
            correction = tf.reduce_sum(correction, axis=1)

        return correction

    def eval(self, observation):
        assert self.built and observation.ndim == 1

        if self._f is None:
            self._f = keras.backend.function(self.inputs, [self.outputs[0]])

        action, = self._f([observation[None]])
        return action.flatten()
