import model
import tensorflow as tf

def loss_bbox_l1_distence(net_out,labels):
    return tf.reduce_sum(tf.abs(tf.subtract(net_out,labels)))

def loss_with_offset(net_out,labels):
    alpa=1
    batch,num=net_out.shape
    cls=net_out[:,0:-1]
    offset=net_out[:,-1]
    labels=tf.minimum(tf.maximum(tf.cast(tf.round(tf.add(offset,tf.cast(labels,tf.float32))),tf.int32),0),3)
    loss_cross_entropy=loss_sparse_softmax_cross_entropy(cls,labels)
    loss=loss_cross_entropy
    return loss

def loss_with_fuzzy_label(net_out,labels):
    softmax_out=tf.nn.softmax(net_out)
    softmax_out=tf.clip_by_value(softmax_out,1e-8,1.0)
    label1=tf.one_hot(labels[:, 0], 4)
    label2=tf.one_hot(labels[:,1],4)
    loss1=-tf.reduce_sum(label1 * tf.log(softmax_out), 1)
    loss2=-tf.reduce_sum(label2 * tf.log(softmax_out), 1)
    loss=tf.reduce_min([loss1,loss2],0)
    loss=tf.reduce_mean(loss)
    return loss
    # loss_cross_entropy1 = loss_sparse_softmax_cross_entropy(net_out, labels[:,0])
    # loss_cross_entropy2 = loss_sparse_softmax_cross_entropy(net_out, labels[:,1])
    # loss=tf.reduce_min(tf.concat([loss_cross_entropy1,loss_cross_entropy2],1))
    # return loss

def loss_with_offset_fuzzy(net_out,labels):
    alpa=0.5
    beta=0.5
    theta=0.1
    cls = net_out[:, 0:-1]
    offset = net_out[:, -1]
    loss_cls=loss_with_fuzzy_label(cls,labels)
    labels=tf.cast(labels,tf.float32)
    mean_label=tf.reduce_mean(labels,1)
    loss_offset=tf.abs(tf.subtract(tf.add(tf.cast(tf.argmax(cls,1),tf.float32),offset),mean_label))
    loss_offset_value=tf.abs(offset)
    loss=tf.multiply(alpa,loss_cls)+tf.multiply(beta,loss_offset)+tf.multiply(theta,loss_offset_value)
    loss=tf.reduce_mean(loss)
    return loss

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

def evaluation_with_offfset(net_out,labels):
    cls = net_out[:, 0:-1]
    offset = net_out[:, -1]
    labels = tf.minimum(tf.maximum(tf.cast(tf.round(tf.add(offset, tf.cast(labels, tf.float32))), tf.int32), 0), 3)
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(cls,labels,1)
        correct = tf.cast(correct,tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return accuracy
def evaluationwith_fuzzy_label(net_out,labels):
    correct1 = tf.nn.in_top_k(net_out, labels[:,0], 1)
    correct2 = tf.nn.in_top_k(net_out, labels[:,1], 1)
    correct = tf.logical_or(correct1, correct2)
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct)
    return accuracy
def evaluationwith_offset_fuzzy_label(net_out,labels):
    cls = net_out[:, 0:-1]
    offset = net_out[:, -1]
    labels=tf.cast(labels,tf.float32)
    mean_label=tf.reduce_mean(labels,1)
    correct=tf.less(tf.abs(tf.subtract(tf.add(tf.cast(tf.argmax(cls, 1), tf.float32), offset),mean_label)),0.5)
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct)
    return accuracy

def evaluationwith_offset_fuzzy_label1(net_out,labels):
    cls = net_out[:, 0:-1]
    offset = net_out[:, -1]
    label1 = labels[:, 0]
    label2 = labels[:, 1]
    accuracy1 = tf.equal(tf.cast(tf.round(tf.add(tf.cast(tf.argmax(cls, 1), tf.float32), offset)),tf.int32),label1)
    accuracy2 = tf.equal(tf.cast(tf.round(tf.add(tf.cast(tf.argmax(cls, 1), tf.float32), offset)),tf.int32),label2)
    accuracy=tf.logical_or(accuracy1,accuracy2)
    accuracy=tf.cast(accuracy,tf.float32)
    accuracy=tf.reduce_mean(accuracy)
    return accuracy