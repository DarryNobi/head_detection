import numpy as np
import tensorflow as tf
import os
import time
import ops
import data_buscrowd
import matplotlib.pyplot as plt
# from model import model
# from networks.mymodel_with_offset import model as model
from networks.resnet import resnet_v2_152 as  model
# from networks.google_v3 import inception_v3 as model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '5,7'

MAX_STEP=15000
BATCH_SIZE=32
logs_train_dir='output/'

def save_list(data_list,name='result.txt'):
    fp = open(name, 'w+')
    for i in range(len(data_list)):
        fp.write(str(data_list[i]))
        fp.write(" ")
    fp.close()

def remove_output():
    path = 'output/'
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

def check_weights(sess,scope_name,tensor_name):
    with tf.variable_scope(scope_name,reuse=True):
        tensor = tf.get_variable(tensor_name)
        val=sess.run(tensor)
        print(tensor_name,val.shape,val)
    return val

def train():
    remove_output()
    is_training = tf.placeholder(tf.bool, shape=())
    train_img, train_lable = data_buscrowd.get_train_data(batch_size=BATCH_SIZE)
    test_img, test_lable = data_buscrowd.get_test_data(batch_size=BATCH_SIZE)
    x = tf.cond(is_training, lambda: train_img, lambda: test_img)
    y = tf.cond(is_training, lambda: train_lable, lambda: test_lable)
    logits=model(x,num_classes=4)
    losses=ops.loss_sparse_softmax_cross_entropy(logits,y)
    acc=ops.evaluation(logits,y)
    # learing_rate = tf.train.exponential_decay(
    #     learning_rate=0.01, global_step=MAX_STEP, decay_steps=200, decay_rate=0.1, staircase=True)

    train_op=ops.optimize_adam(losses,0.01)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        max_acc=0
        try:
            total_loss=[]
            total_acc=[]
            total_test_acc=[]
            plt.ion()  # 开启interactive mode 成功的关键函数
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                _,val_loss,val_acc,val_logits,val_lable = sess.run([train_op,losses,acc,logits,y],feed_dict={is_training: True})
                if step % 10 == 0:
                    val_loss_test, val_acc_test = sess.run([losses, acc], feed_dict={is_training: False})
                    total_loss.append(val_loss)
                    total_acc.append(val_acc)
                    total_test_acc.append(val_acc_test)
                    plt.subplot(211)
                    plt.plot(total_loss,'-r')
                    plt.subplot(212)
                    plt.plot(total_acc,'-y')
                    plt.plot(total_test_acc,'-g')
                    plt.pause(0.1)
                # check_weights( sess,'conv1','weight')
                if step % 500 == 0 or (step + 1) == MAX_STEP:
                    print('\n',step,' loss:',val_loss)
                    print('     ','acc:',val_acc)
                    print('eg:',val_logits[0],val_lable[0])
                    val_logits=np.argmax(val_logits,1)
                    print(val_logits-val_lable)
                    if(val_acc>max_acc):
                        max_acc=val_acc
                        checkpoint_path = os.path.join(logs_train_dir, 'model_'+str(max_acc)+'.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
            save_list(total_acc,'acc_offset.txt')
            save_list(total_test_acc,'test_acc_offset.txt')
            save_list(total_loss,'loss_offset.txt')
        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

def test():
    log_dir='output/'
    img, lable = data_buscrowd.get_test_data(batch_size=BATCH_SIZE)
    x = img
    y = lable
    x = tf.cast(x, tf.float32)
    logits = model(x, num_classes=5)
    losses = ops.loss_with_offset(logits, y)
    acc = ops.evaluation_with_offfset(logits, y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint')
        acc_total=[]
        cls_matrix=np.zeros([4,4])
        try:
            for i in range(20):
                val_loss,val_acc,val_logits,val_y = sess.run([losses,acc,logits,y])
                max_index = np.argmax(val_logits[:,0:-1],1)
                for j in range(32):
                    cls_matrix[val_y[j],max_index[j]]+=1
                acc_total.append(val_acc)
                print(val_acc)
        except tf.errors.OutOfRangeError:
            print('Done test epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
        return np.average(acc_total),cls_matrix

if __name__=='__main__':
    train()
    # print(test())