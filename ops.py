import model
import tensorflow as tf

def loss_bbox_l1_distence(net_out,labels):
    return tf.reduce_sum(tf.abs(tf.subtract(net_out,labels)))

def loss_bbox_l1_distence(net_out,labels):
    return tf.reduce_sum(tf.abs(tf.subtract(net_out,labels)))

def loss_sparse_softmax_cross_entropy(logits,labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits,labels = labels,name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy,name = 'loss')
    return loss
def optimize_adam(loss,learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        global_step = tf.Variable(0,name = 'global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(net_out,labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(net_out,labels,1)
        correct = tf.cast(correct,tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy