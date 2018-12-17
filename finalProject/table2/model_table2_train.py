import numpy as np 
import os 
import tensorflow as tf 
import time
from datetime import datetime
from model_table2 import Model

import sys 
sys.path.append(os.path.dirname(os.getcwd()))

from utils import utils

TRAIN_SAMPLES = 40000 
VAL_SAMPLES = 10000

##############################
# Flags most related to you #
##############################
tf.app.flags.DEFINE_integer(
    'batch_size', 64, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'epoch_number', 50,
    'Number of epoches')

tf.app.flags.DEFINE_string(
    'data_dir', None,
    'Directory of dataset.')

tf.app.flags.DEFINE_string(
    'train_dir', None,
    'Directory where checkpoints and event logs are written to.')

##############################
# Flags for learning rate #
##############################
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum for MomentumOptimizer.')

tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')

##############################
# Flags for log and summary #
##############################
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 30,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'summary_every_n_steps', 30,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 300,
    'The frequency with which the model is saved, in seconds.')

FLAGS = tf.app.flags.FLAGS

##############################
#       Build ResNet         #
##############################
images, labels = utils.get_data(FLAGS.data_dir, 'train', FLAGS.batch_size)

model_table2 = Model(num_classes=10, data_format='channels_first')

initial_conv = [64, 3, 1]
configuration = [
    {
      'name': 'block_layer1',
      'filters': [[64, 64, 128]] * 2,
      'kernel_sizes': [[3, 3, 1]] * 2,
      'strides': [[1, 1, 2], [1, 1, 1]]},
    {
      'name': 'block_layer2',
      'filters': [[64, 128, 128]] * 3,
      'kernel_sizes': [[1, 3, 3]] * 3,
      'strides': [[1, 2, 1]] + [[1, 1, 1]] * 2}, 
    {
      'name': 'block_layer3',
      'filters': [[128, 128, 128]] * 4,
      'kernel_sizes': [[1, 3, 3]] * 4,
      'strides': [[1, 1, 1]] * 4},
    {
      'name': 'block_layer4',
      'filters': [[512, 512]] * 2,
      'kernel_sizes': [[3, 3]] * 2,
      'strides': [[2, 1], [1, 1]]}]

############################################
# Loss, Accuracy, Train, Summary and Saver #
############################################
weight_decay = 2e-4

logits = model_table2(images, initial_conv, configuration, training=True, show=True)

cross_entropy = utils.get_cross_entropy(logits, labels)
accuracy = utils.get_accuracy(logits, labels)
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)

reg_loss = utils.get_reg_loss(weight_decay)
tf.summary.scalar('reg_loss', reg_loss)

total_loss = cross_entropy + reg_loss
tf.summary.scalar('total_loss', total_loss)

global_step = tf.train.create_global_step()
learning_rate = utils.configure_learning_rate(global_step, TRAIN_SAMPLES, FLAGS)
tf.summary.scalar('learning_rate', learning_rate)

optimizer = tf.train.MomentumOptimizer(
    learning_rate=learning_rate,
    momentum=FLAGS.momentum)
grads = optimizer.compute_gradients(total_loss)
train_op = optimizer.apply_gradients(grads, global_step=global_step)
summary_op = tf.summary.merge_all()

saver = tf.train.Saver(tf.global_variables())

############################################
#           For   validation               #
############################################
var_exclude = [v.name for v in tf.local_variables()]
images_val, labels_val = utils.get_data(FLAGS.data_dir, 'validation', FLAGS.batch_size)
logits_val = model_table2(images_val, initial_conv, configuration, training=False)
accuracy_val = utils.get_accuracy(logits_val, labels_val)

# clear former accuracy information for validation
var_to_refresh = [v for v in tf.local_variables() if v.name not in var_exclude]
init_local_val = tf.variables_initializer(var_to_refresh)

############################################
#           Using    Session               #
############################################
sess = tf.Session()

init_global = tf.global_variables_initializer() 
init_local = tf.local_variables_initializer()
train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/log', sess.graph)

# update trainable variables in the graph
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = tf.group(train_op, update_ops)

sess.run(init_global)
sess.run(init_local)

############################################
#           Let's start running            #
############################################
epoch_steps = int(TRAIN_SAMPLES / FLAGS.batch_size)
print('Number of steps for each epoch: ', epoch_steps)
epoch_index = 0
max_steps = FLAGS.epoch_number * epoch_steps
ori_time = time.time()
next_save_time = FLAGS.save_interval_secs
for step in range(max_steps):
    start_time = time.time()
    if step % epoch_steps == 0:
        epoch_index += 1
        if epoch_index > 0:
          sess.run(init_local_val)
          accuracy_val_value = utils.validate(sess, accuracy_val, FLAGS.batch_size, VAL_SAMPLES)
          duration = time.time() - start_time
          duration = float(duration) / 60.0
          val_format = 'Time of validation after epoch %02d: %.2f mins, val accuracy: %.4f'
          print(val_format % (epoch_index - 1, duration, accuracy_val_value))
          

    [_, total_l_value, entropy_l_value, reg_l_value, acc_value] = \
        sess.run([train_op, total_loss, cross_entropy, reg_loss, accuracy])

    total_duration = time.time() - ori_time
    total_duration = float(total_duration)

    assert not np.isnan(total_l_value), 'Model diverged with loss = NaN' 

    if step % FLAGS.log_every_n_steps == 0:
      format_str = ('Epoch %02d/%2d time=%.2f mins: step %d total loss=%.4f loss=%.4f reg loss=%.4f accuracy=%.4f')
      print(format_str % (epoch_index, FLAGS.epoch_number, total_duration / 60.0, step, total_l_value, entropy_l_value, reg_l_value, acc_value))

    if step % FLAGS.summary_every_n_steps == 0:
      summary_str = sess.run(summary_op)
      train_writer.add_summary(summary_str, step)

    if total_duration > next_save_time:
      next_save_time += FLAGS.save_interval_secs
      checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      save_path = saver.save(sess, checkpoint_path, global_step=global_step)
      print('saved model to %s' % save_path)

checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
save_path = saver.save(sess, checkpoint_path, global_step=global_step)
print('saved the final model to %s' % save_path)

sess.run(init_local_val)
accuracy_val_value = utils.validate(sess, accuracy_val, FLAGS.batch_size, VAL_SAMPLES)
print('validation accuracy of the final model: %.4f' % accuracy_val_value)

