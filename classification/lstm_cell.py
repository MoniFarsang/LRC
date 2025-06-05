# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.
# https://github.com/mlech26l/ode-lstms/blob/master/node_cell.py

import tensorflow as tf

class LSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = (units, units)
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        super(LSTMCell, self).__init__(**kwargs)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_shape = (input_shape[0][-1] + input_shape[1][-1],)

        self.input_kernel = self.add_weight(
            shape=(input_shape[-1], 4 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(4 * self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )

        self.built = True

    def call(self, inputs, states):
        cell_state, output_state = states
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            inputs = tf.concat([inputs[0], inputs[1]], axis=-1)

        z = (
            tf.matmul(inputs, self.input_kernel)
            + tf.matmul(output_state, self.recurrent_kernel)
            + self.bias
        )
        i, ig, fg, og = tf.split(z, 4, axis=-1)

        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg + 1.0)
        output_gate = tf.nn.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = tf.nn.tanh(new_cell) * output_gate

        return output_state, [new_cell, output_state]