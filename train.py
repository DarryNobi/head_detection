from PIL import Image
import numpy as np
import tensorflow as tf
import os
import model
import ops
import data_brainwash
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

MAX_STEP=100
logs_train_dir='output/'
def train():
    data_fun=data_brainwash.get_image()
    img_path,rects=data_fun.__next__()
    image = Image.open(os.path.join('data/brainwash/',img_path))
    # plt.imshow(image)
    image_arr = np.array(image)
    image_arr=image_arr.reshape([1,640,480,3])

    x=tf.cast(image_arr,tf.float32)
    y=int(rects[0]['x1'])

    logits=model.model(x)
    losses=tf.cast(ops.loss(logits,y),tf.float32)
    train_op=ops.trainning(losses,0.1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                _ = sess.run(train_op)
                if step % 50 == 0 or (step + 1) == MAX_STEP:
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