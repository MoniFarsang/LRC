import tensorflow as tf

class MGUCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MGUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_shape = (input_shape[0][-1] + input_shape[1][-1],)
            
        self.forget_kernel = self.add_weight(
            shape=(input_shape[-1] + self.units, self.units),
            initializer="glorot_uniform",
            name="forget_kernel",
        )
        
        self.hidden_kernel = self.add_weight(
            shape=(input_shape[-1] + self.units, self.units),
            initializer="glorot_uniform",
            name="hidden_kernel",
        )
        
        self.forget_bias = self.add_weight(
            shape=(self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="forget_bias",
        )
        
        self.hidden_bias = self.add_weight(
            shape=(self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="hidden_bias",
        )
        
        self.built = True

    def call(self, inputs, states):
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            inputs = tf.concat([inputs[0], inputs[1]], axis=-1)
            
        h_prev = states[0]
        
        fused_input = tf.concat([inputs, h_prev], axis=-1)
        
        fg = tf.nn.sigmoid(tf.matmul(fused_input, self.forget_kernel) + self.forget_bias)
        
        h_tilde = tf.nn.tanh(tf.matmul(tf.concat([inputs, fg * h_prev], axis=-1), self.hidden_kernel) + self.hidden_bias)
        
        ht = (1 - fg) * h_prev + fg * h_tilde
        return ht, [ht]
    
    
class GRUCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(GRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape[0], tuple):
            # Nested tuple
            input_shape = (input_shape[0][-1] + input_shape[1][-1],)
                    
        self.update_kernel = self.add_weight(
            shape=(input_shape[-1] + self.units, self.units),
            initializer="glorot_uniform",
            name="update_kernel",
        )
        
        self.reset_kernel = self.add_weight(
            shape=(input_shape[-1] + self.units, self.units),
            initializer="glorot_uniform",
            name="reset_kernel",
        )
        
        self.hidden_kernel = self.add_weight(
            shape=(input_shape[-1] + self.units, self.units),
            initializer="glorot_uniform",
            name="hidden_kernel",
        )
        
        self.update_bias = self.add_weight(
            shape=(self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="update_bias",
        )
        
        self.reset_bias = self.add_weight(
            shape=(self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="reset_bias",
        )
            
        self.hidden_bias = self.add_weight(
            shape=(self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="hidden_bias",
        )
        
        self.built = True

    def call(self, inputs, states):
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            inputs = tf.concat([inputs[0], inputs[1]], axis=-1)
            
        h_prev = states[0]
        
        fused_input = tf.concat([inputs, h_prev], axis=-1)
        
        zg = tf.nn.sigmoid(tf.matmul(fused_input, self.update_kernel) + self.update_bias)
        
        rg = tf.nn.sigmoid(tf.matmul(fused_input, self.reset_kernel) + self.reset_bias)
        
        h_tilde = tf.nn.tanh(tf.matmul(tf.concat([inputs, rg * h_prev], axis=-1), self.hidden_kernel) + self.hidden_bias)
        
        ht = (1 - zg) * h_prev + zg * h_tilde
        return ht, [ht]