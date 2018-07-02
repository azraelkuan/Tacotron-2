import tensorflow as tf


class Conv1d1x1(object):

    def __init__(self, filters, kernel_size, dilation, use_bias=True, padding=None, scope=None, name='conv1d1x1'):
        self._linearized_weight = None

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_bias = use_bias
        self.padding = padding
        self.name = name
        self.scope = scope

        self.conv = tf.layers.Conv1D(filters, kernel_size, dilation_rate=dilation, use_bias=use_bias, name=name)

    def __call__(self, inputs):
        if self.padding is not None:
            inputs = tf.pad(inputs, tf.constant([(0, 0,), (self.padding, 0), (0, 0)]))
        outputs = self.conv(inputs)
        return outputs

    def incremental_forward(self, inputs, input_buffer):
        input_shape = inputs.get_shape().as_list()

        # reshape weight
        weight = self._get_linearized_weight(inputs)
        kw = self.kernel_size
        dilation = self.dilation

        batch_size = input_shape[0]
        if kw > 1:
            input_buffer = input_buffer[:, 1:, :]
            input_buffer = tf.concat([input_buffer, tf.expand_dims(inputs[:, -1, :], 1)], axis=1)
            inputs = tf.Print(input_buffer, [input_buffer], 'Debug:')
            if dilation > 1:
                inputs = inputs[:, 0::dilation, :]

        inputs = tf.reshape(inputs, [batch_size, -1])
        output = tf.matmul(inputs, weight, transpose_b=True)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.conv.bias)
        return tf.reshape(output, [batch_size, 1, self.filters]), input_buffer

    def _get_linearized_weight(self, inputs):
        current_scope = "{}/{}".format(tf.get_variable_scope().name, self.name)
        print(current_scope)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=current_scope)
        for i in variables:
            print(i.name)
        with tf.control_dependencies([tf.assert_equal(len(variables), 2)]):

            weight = variables[0]
            self.bias = variables[1]

            input_shape = inputs.get_shape().as_list()
            input_channel = input_shape[-1]

            if weight.shape == (self.filters, input_channel, self.kernel_size):
                weight = tf.transpose(weight, [0, 2, 1])
            else:
                # layers.Conv1D kw, in_channel, filters
                weight = tf.transpose(weight, [2, 0, 1])

            with tf.control_dependencies([tf.assert_equal(weight.shape,
                                                          (self.filters, self.kernel_size, input_channel))]):
                linearized_weight = tf.cast(tf.reshape(weight, [self.filters, -1]), dtype=inputs.dtype)
                return linearized_weight

