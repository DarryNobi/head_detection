import tensorflow as tf

def model(input,num_classes):
    batch_size=input.shape[0]
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable('weight',
                                  shape=[3, 3, 3, 10],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[10],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input, weights, strides=[1, 2, 2, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_out = tf.nn.relu(pre_activation,scope.name)
    with tf.variable_scope('pooling1') as scope:
        pool1_out = tf.nn.max_pool(conv1_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling')
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable('weight',
                                  shape=[3, 3, 10, 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1_out, weights, strides=[1, 2, 2, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_out = tf.nn.relu(pre_activation,scope.name)
    with tf.variable_scope('pooling2') as scope:
        pool2_out = tf.nn.max_pool(conv2_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling')

    with tf.variable_scope("conv3") as scope:
        weights = tf.get_variable('weight',
                                  shape=[3, 3, 32, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2_out, weights, strides=[1, 2, 2, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3_out = tf.nn.relu(pre_activation,scope.name)
    with tf.variable_scope('pooling2') as scope:
        pool3_out = tf.nn.max_pool(conv3_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling')

    reshape = tf.reshape(pool3_out, shape=[batch_size, -1])
    dim = reshape.get_shape()[1].value

    with tf.variable_scope('fc1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[dim, 100],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[100],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        fc1_out = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='local4')

    with tf.variable_scope('fc2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[100, num_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[num_classes],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        fc2_out = tf.nn.relu(tf.matmul(fc1_out, weights) + biases, name='local4')

    output=fc2_out
    return output
