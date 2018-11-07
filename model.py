import tensorflow as tf

def model(input):
    with tf.variable_scope("conv1"):
        weights = tf.get_variable('weight',
                                  shape=[3, 3, 3, 1],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[1],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    relu = tf.nn.relu(pre_activation)
    output=tf.reduce_mean(relu)
    return output
