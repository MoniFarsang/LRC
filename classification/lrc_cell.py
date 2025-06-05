# Follows the LTC implementation by Mathias Lechner and Ramin Hasani (2022) https://github.com/mlech26l/ncps/blob/master/ncps/tf/ltc_cell.py

import tensorflow as tf

class LRC_Cell(tf.keras.layers.AbstractRNNCell):
    def __init__(
        self,
        units,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=1,
        dt = 1,
        elastance_type = "interp",
        forget_gate = True,
        ode_solver = 'explicit',
        epsilon=1e-8,
        initialization_ranges=None,
        **kwargs
    ):
        """
            Liquid-Resistance Liquid-Capacitance (LRC) <https://arxiv.org/pdf/2403.08791> cell.
            It extends the Liquid Time-Constant (LTC) cell <https://arxiv.org/abs/2002.05202> by adding a saturation functions and input- and state-dependent elastance.

            This is an AbstractRNNCell that process single time-steps.
            To get a full RNN that can process sequences, it needs to be wrapped by a tf.keras.layers.RNN <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN>.

             >>> cell = LRC_Cell(units)
             >>> rnn = tf.keras.layers.RNN(cell)

        """

        super().__init__(**kwargs)
        self.units = units
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        if not initialization_ranges is None:
            for k, v in initialization_ranges.items():
                if k not in self._init_ranges.keys():
                    raise ValueError(
                        "Unknown parameter '{}' in initialization range dictionary! (Expected only {})".format(
                            k, str(list(self._init_range.keys()))
                        )
                    )
                if k in ["gleak", "w", "sensory_w"] and v[0] < 0:
                    raise ValueError(
                        "Initialization range of parameter '{}' must be non-negative!".format(
                            k
                        )
                    )
                if v[0] > v[1]:
                    raise ValueError(
                        "Initialization range of parameter '{}' is not a valid range".format(
                            k
                        )
                    )
                self._init_ranges[k] = v

        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._dt = dt
        self._elastance_type = elastance_type
        self._forget_gate = forget_gate
        self._epsilon = epsilon
        self._ode_solver_type = ode_solver
        
        self._layerwise = False

    @property
    def state_size(self):
        return self.units

    @property
    def sensory_size(self):
        return self.input_dim

    def _get_initializer(self, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return tf.keras.initializers.Constant(minval)
        else:
            return tf.keras.initializers.RandomUniform(minval, maxval)

    def build(self, input_shape):
        # Check if input_shape is nested tuple/list
        if isinstance(input_shape[0], tuple) or isinstance(
            input_shape[0], tf.TensorShape
        ):
            # Nested tuple -> First item represent feature dimension
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]
            
        self.input_dim = input_dim

        self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak",
            shape=(self.state_size,),
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg(),
            initializer=self._get_initializer("gleak"),
        )
        self._params["vleak"] = self.add_weight(
            name="vleak",
            shape=(self.state_size,),
            dtype=tf.float32,
            initializer=self._get_initializer("vleak"),
        )
        self._params["sigma"] = self.add_weight(
            name="sigma",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sigma"),
        )
        self._params["mu"] = self.add_weight(
            name="mu",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("mu"),
        )
        if self._forget_gate:
            self._params["w"] = self.add_weight(
                name="w",
                shape=(self.state_size, self.state_size),
                dtype=tf.float32,
                constraint=tf.keras.constraints.NonNeg(),
                initializer=self._get_initializer("w"),
            )
            self._params["sensory_w"] = self.add_weight(
                name="sensory_w",
                shape=(self.sensory_size, self.state_size),
                dtype=tf.float32,
                constraint=tf.keras.constraints.NonNeg(),
                initializer=self._get_initializer("sensory_w"),
            )
        else:
            self._params["tau"] = self.add_weight(
                name="tau",
                shape=(self.state_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1),
                constraint=tf.keras.constraints.NonNeg()
            )
        
        self._params["h"] = self.add_weight(
            name="h",
            shape=(self.state_size, self.state_size),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Orthogonal(),
        )
        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sensory_sigma"),
        )
        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=self._get_initializer("sensory_mu"),
        )

        self._params["sensory_h"] = self.add_weight(
            name="sensory_h",
            shape=(self.sensory_size, self.state_size),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Orthogonal(),
        )

        self.elastance_mapping = tf.keras.layers.Dense(self.state_size, name="elastance_mapping")
            
        if self._elastance_type in ["symmetric"]:
            self._params["distr_shift"] = self.add_weight(
                name="distr_shift",
                shape=(self.state_size,),
                dtype=tf.float32,
                constraint=tf.keras.constraints.NonNeg(),
                initializer=tf.keras.initializers.Constant(1)
            )

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                shape=(self.sensory_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1),
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                shape=(self.sensory_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(0),
            )

        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                shape=(self.state_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1),
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                shape=(self.state_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(0),
            )


        self._prev_dts = [self._dt] * self.state_size

        self.built = True

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = tf.expand_dims(v_pre, axis=-1)  # For broadcasting
        mues = v_pre - mu
        x = sigma * mues
        return tf.nn.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_syn = self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )

        sensory_h_activation = self._params["sensory_h"] * sensory_syn

        # Reduce over dimension 1 (=source sensory neurons)
        if self._forget_gate:
            sensory_w_activation = self._params["sensory_w"] * sensory_syn
            sensory_w_activation_reduced = tf.reduce_sum(sensory_w_activation, axis=1)
        sensory_h_activation_reduced = tf.reduce_sum(sensory_h_activation, axis=1)

        dt = elapsed_time / self._ode_unfolds

        # Unfold the multiply ODE multiple times into one RNN step
        for t in range(self._ode_unfolds): # 1 unfold is enough for LRC, but leaving this here to experiment with more unfolds
            if self._elastance_type == "asymmetric":
                x = tf.keras.layers.Concatenate()([inputs, v_pre])
                elast_dense = self.elastance_mapping(x)
                elastance_t = tf.nn.sigmoid(elast_dense) * dt 
            elif self._elastance_type == "symmetric":
                x = tf.keras.layers.Concatenate()([inputs, v_pre])
                elast_dense = self.elastance_mapping(x)
                elastance_t = (tf.nn.sigmoid(elast_dense + self._params["distr_shift"]) - tf.nn.sigmoid(elast_dense - self._params["distr_shift"])) * dt 
            else:
                elastance_t = dt
        
            syn = self._sigmoid(
                v_pre, self._params["mu"], self._params["sigma"]
            )

            h_activation = self._params["h"] * syn

            g = self._params["gleak"] + tf.reduce_sum(h_activation, axis=1) + sensory_h_activation_reduced

            if self._forget_gate:
                w_activation = self._params["w"] * syn
                f = self._params["gleak"] + tf.reduce_sum(w_activation, axis=1) + sensory_w_activation_reduced
                v_prime = - v_pre * tf.nn.sigmoid(f)  + self._params["vleak"]*tf.nn.tanh(g)
            else:
                v_prime = - v_pre * self._params["tau"] + self._params["vleak"]*tf.nn.tanh(g)

            if self._ode_solver_type == 'hybrid':
                v_pre = (elastance_t * self._params["vleak"] * tf.nn.tanh(g) + v_pre)/(1+elastance_t * tf.nn.sigmoid(f))
            else:
                v_pre = v_pre + elastance_t * v_prime
        
        return v_pre

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state
        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            # Irregularly sampled mode
            inputs, elapsed_time = inputs
        else:
            # Regularly sampled mode (elapsed time = 1 second)
            elapsed_time = self._dt
        inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(inputs, states[0], elapsed_time)

        outputs = self._map_outputs(next_state)

        return outputs, [next_state]