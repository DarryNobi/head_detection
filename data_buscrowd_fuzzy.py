import csv
import tensorflow as tf
import numpy as np
def get_file_dir(file):
    image_dir = []
    label_dir1 = []
    label_dir2 = []
    r_csv = open(file)
    path_lists = list(csv.reader(r_csv))
    for i in path_lists:
        image_dir.append(i[0])
        label_dir1.append(i[1])
        label_dir2.append(i[1])
    label_dir1 = [int(j) for j in label_dir1]
    label_dir2 = [int(j) for j in label_dir2]
    return image_dir,label_dir1,label_dir2

def get_batch(image,label1,label2,batch_size=1,capacity=0,image_W=0,image_H=0):
    if capacity==0:
        capacity=len(image)
    image = tf.cast(image,tf.string)
    label=[label1,label2]
    label=np.transpose(label)
    label = tf.cast(label,tf.int32)
    input_queue = tf.train.slice_input_producer([image,label],shuffle=True)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    if (image_W*image_H!=0):
        image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    else:
        image=tf.image.resize_image_with_crop_or_pad(image, 299, 299)
    image = tf.cast(image,tf.float32)
    image = tf.image.per_image_standardization(image)
    image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads=16,capacity = capacity)
    image_batch=tf.cast(image_batch,tf.float32)
    label_batch = tf.reshape(label_batch,[batch_size,2])
    return image_batch,label_batch

def get_train_data(batch_size):
    train_file_dir, train_label1,train_label2 = get_file_dir("data/buscrowd/train.csv")
    train_batch, train_label_batch = get_batch(train_file_dir, train_label1,train_label2,batch_size=batch_size)
    return train_batch, train_label_batch

def get_test_data(batch_size):
    test_file_dir, test_label1,test_label2 = get_file_dir("data/buscrowd/test.csv")
    test_batch, test_label_batch = get_batch(test_file_dir, test_label1,test_label2,batch_size=batch_size)
    return test_batch, test_label_batch


