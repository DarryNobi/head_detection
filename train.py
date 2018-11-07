import numpy as np
import tensorflow as tf
import os
import ops
import data_buscrowd
# from model import model
from networks.resnet import resnet_v2_152 as  model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

MAX_STEP=10000
BATCH_SIZE=32
logs_train_dir='output/'

def check_weights(sess,scope_name,tensor_name):
    with tf.variable_scope(scope_name,reuse=True):
        tensor = tf.get_variable(tensor_name)
        val=sess.run(tensor)
        print(tensor_name,val.shape,val)
        return val

def train():
    img,lable=data_buscrowd.get_test_data(batch_size=BATCH_SIZE)
    x=img
    y=lable

    x=tf.cast(x,tf.float32)
    logits=model(x,num_classes=4)
    losses=ops.loss_sparse_softmax_cross_entropy(logits,y)
    acc=ops.evaluation(logits,y)
    train_op=ops.optimize_adam(losses,0.0001)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                _,val_loss,val_acc,val_logits,val_lable = sess.run([train_op,losses,acc,logits,y])
                # check_weights(sess,'conv1','weight')
                if step % 50 == 0 or (step + 1) == MAX_STEP:
                    print('\n',step,' loss:',val_loss)
                    print('     ','acc:',val_acc)
                    val_logits=np.argmax(val_logits,1)
                    print(val_logits-val_lable)
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__=='__main__':
    train()