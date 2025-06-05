import tensorflow as tf
from neuralODE.lrc_ar_cell import LRC_AR_Cell

# Build model with LRC_Cell    
class ODEFunc(tf.keras.Model):
    def __init__(self, input_shape, elastance, units, **kwargs):
        super(ODEFunc, self).__init__(**kwargs)

        input = tf.keras.Input(shape=input_shape, name="features", dtype = tf.float32)

        dense_layer_in = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units))

        cell = LRC_AR_Cell(units=units, elastance_type=elastance, output_mapping=None, input_mapping=None)
        rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=True)

        dense_layer_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_shape[-1]))

        output_states = rnn((dense_layer_in(input)))

        y = dense_layer_out(output_states)

        self.model = tf.keras.Model(inputs=[input], outputs=[y])

    def call(self, t, y):
        y = self.model(y)
        return y

# Build standard model
class NODEFunc(tf.keras.Model):
    def __init__(self, input_shape, **kwargs):
        super(NODEFunc, self).__init__(**kwargs)
        self.x = tf.keras.layers.Dense(32, activation='tanh',
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.z = tf.keras.layers.Dense(32, activation='tanh',
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        self.y = tf.keras.layers.Dense(input_shape[-1],
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

    def call(self, t, y0):
        x = self.x(y0)
        z = self.z(x)
        y = self.y(z)
        return y