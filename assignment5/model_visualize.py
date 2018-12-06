import tensorflow as tf 
from datasets import cifar10
from mymodel import *
import os 
import time
from datetime import datetime
from visual import conviz

slim = tf.contrib.slim 

# set the model and data path here
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to and load from.')

tf.app.flags.DEFINE_string(
    'data_dir', '/tmp/tfdata/',
    'Directory of dataset.')

tf.app.flags.DEFINE_string(
    'visual_dir', '/tmp/tfvisual/',
    'Directory of visualization results.')


FLAGS = tf.app.flags.FLAGS

# prepare images and lables for cifar10 dataset
dataset = cifar10.get_split('test', FLAGS.data_dir)
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])

images = tf.expand_dims(image, 0)
labels = tf.expand_dims(label, 0)

labels = slim.one_hot_encoding(
  labels, dataset.num_classes)

images = tf.cast(images, tf.float32)
# rerange images to [-4, 4] as the training did
images = (images - 127) / 128 * 4

model = Mymodel(dropout=1.0)
model.build(images, labels, train_mode=False)

init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

sess = tf.Session()

variables_to_restore = tf.global_variables()
restorer = tf.train.Saver(variables_to_restore)

checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
restorer.restore(sess, os.path.join(FLAGS.train_dir, checkpoint_path))

sess.run(init_local)

tf.train.start_queue_runners(sess=sess)
visual_target = [model.conv_l_5, 
        model.conv_l_11, 
        model.draw_filter_5, 
        model.draw_filter_11]

visual_value = sess.run(visual_target)

# begin to visualize conv layers and filters
# OK, I can not help you more, please add your own code here

print('Please add your own code')
