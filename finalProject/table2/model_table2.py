from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
    kernel_sizes, data_format, show):
  """A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: A list of filter_number in the block
    strides: A list of strides in the block
    kernel_sizes: A list of kernel_size in the block
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    if show:
      print('Shortcut for the follwing block element: ', shortcut.get_shape())
  else:
    if show:
      print('Direct shorcut for the follwing block element')

  for i in range(len(filters)):
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters[i], 
        kernel_size=kernel_sizes[i], strides=strides[i], data_format=data_format)
    if i < (len(filters) - 1):
      inputs = batch_norm(inputs, training, data_format)
      inputs = tf.nn.relu(inputs)

    if show:
      print('After %dth conv layer of the current block element: ' % (i + 1), inputs.get_shape())

  return inputs + shortcut


def block_layer(inputs, conv_configuration, training, name, data_format, show):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    conv_configuration: A dict contains filters, strides and kernel_sizes, where 
      filters: A list in which each is a list of filter_number for the current block element 
      strides: A list in which each is a list of strides for the current block element
      kernel_sizes: A list in which each is a list of kernel_size for the current element
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  def block_element(inputs, filters, strides, kernel_sizes):
    filters_out = filters[-1]
    strides_out = int(np.max(strides))

    def projection_shortcut(inputs):
      return conv2d_fixed_padding(
          inputs=inputs, filters=filters_out, kernel_size=1, strides=strides_out,
          data_format=data_format)

    if strides_out > 1:
      projection_fn = projection_shortcut 
    else:
      projection_fn = None 

    inputs = _building_block_v2(inputs, filters, training, projection_fn, strides, 
        kernel_sizes, data_format, show=show)
    return inputs

  for i in range(len(conv_configuration['filters'])):
    filters = conv_configuration['filters'][i] 
    strides = conv_configuration['strides'][i] 
    kernel_sizes = conv_configuration['kernel_sizes'][i] 
    inputs = block_element(inputs, filters, strides, kernel_sizes)

    if show:
      print('After %dth block_element: ' % (i + 1), inputs.get_shape(), '\n')

  return tf.identity(inputs, name)


class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self, num_classes, data_format=None):

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.data_format = data_format
    self.num_classes = num_classes
    self.pre_activation = True

  def __call__(self, inputs, initial_conv, configuration, training, show=False):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      initial_conv: A list contains [filter_num, kernel_size, stride]
      configuration: A list in which each is a dict contains filters, strides, kernel_sizes and names, where 
        name: the name of the block layer
        filters: A list in which each is a list of filter_number for the current block element 
        strides: A list in which each is a list of strides for the current block element
        kernel_sizes: A list in which each is a list of kernel_size for the current element
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """
     
    with tf.variable_scope('model_table2', reuse=tf.AUTO_REUSE):

      if show:
        print('Inputs: ', inputs.get_shape())
      inputs = conv2d_fixed_padding(
          inputs=inputs, filters=initial_conv[0], kernel_size=initial_conv[1],
          strides=initial_conv[2], data_format=self.data_format)
      inputs = tf.identity(inputs, 'initial_conv')
      if show:
        print('After initial conv: ', inputs.get_shape(), '\n\n')

      for conv_config in configuration:
        inputs = block_layer(inputs, conv_config, training=training, 
            name=conv_config['name'], data_format=self.data_format, show=show)
        if show:
          print('After ', conv_config['name'], ': ', inputs.get_shape(), '\n\n')

      if self.pre_activation:
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)

      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(inputs, axes, keepdims=True)
      inputs = tf.identity(inputs, 'final_reduce_mean')

      inputs = tf.squeeze(inputs, axes)
      if show:
        print('After average pooling: ', inputs.get_shape(), '\n\n')

      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
      inputs = tf.identity(inputs, 'final_dense')

      if show:
        print('After FC layer: ', inputs.get_shape(), '\n')

      return inputs
