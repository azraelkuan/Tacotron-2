import numpy as np
import tensorflow as tf
from test_func.test_module import Conv1d1x1



kernel_size = 3
dilation = 1

tmp = tf.placeholder(shape=[1, None, 1], dtype=tf.float32)
time = tf.placeholder(shape=[], dtype=tf.int32)

inputs = np.random.random_sample(10).reshape(1, 10, 1).astype(np.float32)
padding = (kernel_size - 1) * dilation
with tf.variable_scope('conv') as scope:
    conv = Conv1d1x1(filters=1, kernel_size=kernel_size, dilation=dilation, padding=padding, name='conv1d1x1')
    y1 = conv(tmp)

    # y2 = conv.incremental_forward(tmp)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print(tf.global_variables())
    print("inputs")
    for i in range(10):
        print(inputs[:, i, :])

    log1 = sess.run(y1, feed_dict={tmp: inputs})
    print("normal")
    for i in range(10):
        print(log1[:, i, :])
    print("incremental")
    # for i in range(10):
    #     log2, debug1 = sess.run([y2], feed_dict={tmp: np.expand_dims(inputs[:, i, :], 1)})
    #     print(log2)
    #     print(debug1)
    #     # # print(debug2)
    #     # print(np.dot(debug1, debug2))
    # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #     print(i)
    #     print(sess.run(i))

    index = tf.constant(0, dtype=tf.int32)
    outputs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    buffers = [tf.zeros([1, kernel_size + (kernel_size - 1) * (dilation - 1), conv.conv.filters])]
    with tf.variable_scope('conv'):
        def cond(index, test_inputs, outputs, buffers):
            return tf.less(index, 10)

        def body(index, test_inputs, outputs, buffers):
            current_output, buffer = conv.incremental_forward(tf.expand_dims(test_inputs[:, index, ], axis=1), buffers[0])

            buffers[0] = buffer
            outputs = outputs.write(index, current_output)
            index += 1
            return index, test_inputs, outputs, buffers

        result = tf.while_loop(cond, body, loop_vars=[index, inputs, outputs, buffers])
        outputs = result[2].stack()
        print(sess.run(outputs))
