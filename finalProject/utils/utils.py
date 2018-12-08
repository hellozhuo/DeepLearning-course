import tensorflow as tf 
import numpy as np 
import os

def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
      "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  images = parsed_features["image"]
  images = tf.decode_raw(images, tf.uint8)
  # channel first
  images = tf.reshape(images, [3, 32, 32])
  images = tf.cast(images, tf.float32)
  images = (images - 127) / 128.0 * 4
  return images, parsed_features["label"]

def get_data(data_dir, mode, batch_size):
  if mode == 'train':
    file = 'train.tfrecords'
  elif mode == 'validation':
    file = 'validation.tfrecords'
  elif mode == 'eval':
    file = 'eval.tfrecords'
  else:
    raise ValueError('mode should be %s or %s or %s' % ('train', 'validation', 'eval'))

  path = os.path.join(data_dir, file)
  dataset = tf.data.TFRecordDataset(path)
  dataset = dataset.map(_parse_function)

  if mode == 'train':
    dataset = dataset.shuffle(buffer_size=10000)

  dataset = dataset.repeat() 
  dataset = dataset.batch(batch_size) 
  itr = dataset.make_one_shot_iterator()
  images, labels = itr.get_next()
  return images, labels

def configure_learning_rate(global_step, num_samples, FLAGS):
    decay_steps = int(num_samples * FLAGS.num_epochs_per_decay / FLAGS.batch_size)

    return tf.train.exponential_decay(FLAGS.learning_rate,
            global_step,
            decay_steps,
            FLAGS.learning_rate_decay_factor,
            staircase=True,
            name='exponential_decay_learning_rate')

def get_cross_entropy(logits, labels):
  logits = tf.cast(logits, tf.float32)
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
  return cross_entropy

def get_accuracy(logits, labels):
  logits = tf.cast(logits, tf.float32)
  accuracy = tf.metrics.accuracy(labels, tf.argmax(logits, axis=1))
  return accuracy[1]

def get_reg_loss(weight_decay):
  reg_loss = weight_decay * tf.add_n(
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
  return reg_loss

def validate(sess, accuracy_val, batch_size, val_samples):
  num = 1
  while True:
    acc_value = sess.run(accuracy_val)
    num += batch_size
    print('Calculating accuracy on validation set: processed %d samples' % num, end='\r')
    if num > val_samples:
      return acc_value
