import model
import tensorflow as tf

def loss(logits,labels):
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
    #                     (logits=logits,labels = labels,name='xentropy_per_example')
    return tf.subtract(logits,labels)

def trainning(loss,learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        global_step = tf.Variable(0,name = 'global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits,labels,1)
        correct = tf.cast(correct,tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy