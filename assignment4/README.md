# Build own model using tensorflow api 

Let's say, the structure of the model is:

| Input (32 x 32 RGB images) | Layers|
|:----------:|:-------:|
| Conv3-8 | Layer-1 |
| Conv3-8 | Layer-2 |
| Conv3-8 | Layer-3 |
| maxpool | Layer-4 |
| Conv3-64 | Layer-5 |
| Conv3-64 | Layer-6 |
| Conv3-64 | Layer-7 |
| maxpool | Layer-8 |
| Conv3-64 | Layer-9 |
| Conv3-64 | Layer-10 |
| Conv3-64 | Layer-11 |
| maxpool | Layer-12 |
| FC-1024 | Layer-13 |
| FC-10 | Layer-14 |
| softmax | Layer-15 |

* Conv3-8 means the convolutional kernel is 3 × 3, and number of output channels is 8, the padding style is **SAME** rather than **VALID** (see `tf.nn.conv2d`)
* FC-1024 means the output size of the FC layer is 1024
* stride of Conv layers is 1, stride of pooling layers is 2
* kernel size of maxpool is 2 x 2

## Dirs used
Here, we spicify the following paths for the model for convenience. Please change them to your own path when doing yourself.<br>
**Path where you download cifar10 dataset:** */mymodel/cifar10-data*<br>
**Path where you save your model to when training and load your model from when testing or finetuning:** */mymodel/model*<br>
**Path where you save your log informations:** */mymodel/log*<br>

# Prepare cifar10 dataset
As assignment 3, we follow the instructions in [Downloading and converting to TFRecord format](https://github.com/tensorflow/models/tree/master/research/slim) by changing the `DATA_DIR` to '/mymodel/cifar10-data' and `dataset_name` to cifar10<br> 

Now in the /mymodel/cifar10-data, there would be the following files: *cifar10_test.tfrecord  cifar10_train.tfrecord  labels.txt*

# Build your own model with the above structure
Create a file with name "mymodel.py" and build your model there.<br>

First, the input of the model should definitely be the `images`, which has the shape of [batch_size, height, width, channels], i.e., [64, 32, 32, 3] since we now just set the batch size to 64<br> 
We then begin to input the `images` to the structure, the first thing we come across here would be the first Conv3-8 layer, which has kernel size of 3 x 3, stride of 1 and output channels of 8. So when we type:
```python
  conv_l_1 = conv_layer(images, 3, 8, "layer1")
```
we get the output of the first Conv3-8 layer, `conv_l_1`, which has the shape of [batch_size, height_1, width_1, output_channels], i.e., [64, 32, 32, 8]. Since the stride of the conv layer is 1, and the padding strategy is **SAME**, the height and width will not change<br>

Now, take a look at the function `conv_layer`
```python 
  def conv_layer(bottom, in_channels, out_channels, name):
      with tf.variable_scope(name):
          filt, conv_biases = get_conv_var(3, in_channels, out_channels, name)

          conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
          bias = tf.nn.bias_add(conv, conv_biases)
          relu = tf.nn.relu(bias)

          return relu

  def get_conv_var(filter_size, in_channels, out_channels, name):
      filters = tf.get_variable('filters', 
              shape=[filter_size, filter_size, in_channels, out_channels],
              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=5e-2),
              trainable=trainable, # trainable on the right hand is a global variable
              regularizer=None)

      biases = tf.get_variable('biases',
              shape=[out_channels],
              initializer=tf.constant_initializer(value=0.0),
              trainable=trainable,
              regularizer=None)

      return filters, biases
```
where, filters and biases are the parameters of conv layer whose values will be learnt during training, actually, the training process of the model is to find the optimized values of these parameters that minimize the final loss<br>

Similar to layer1, we add two more conv layers, then followed by a maxpool layer. <br> 
```python
  conv_l_2 = conv_layer(conv_l_1, 8, 8, "layer2")
  conv_l_3 = conv_layer(conv_l_2, 8, 8, "layer3")
  pool_l_4 = max_pool(conv_l_3, 'layer4')
```

The functions `max_pool` looks like
```python
  def max_pool(bottom, name):
      return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
```

Like `conv_layer` we just mentioned, it is using tensorflow apis again, i.e., `tf.nn.max_pool`. These are the basic apis in tensorflow for CNN. By following this post, you will get to know many of these basic apis, cheer up!<br> 
OK, now repeat the above process. After some blocks of convs-pools, we get the output of the last max-pooling layer:
```python
  pool_l_12 = max_pool(conv_l_11, 'layer12')
```

A voice says: maybe it's time to connect some FC layers. So we just follow what it says.
```python
  fc_l_13 = fc_layer(pool_l_12, 1024, "layer13")
  fc_l_14 = fc_layer(fc_l_13, 10, "layer14")
```

and fc_layer
```python
  def fc_layer(bottom, out_size, name):
      with tf.variable_scope(name):
          batch_size = bottom.get_shape()[0]
          x = tf.reshape(bottom, [batch_size, -1])
          in_size = x.get_shape()[1]
          weights, biases = get_fc_var(in_size, out_size, name)

          fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
          tf.summary.histogram('weights', weights)

          return fc

  def get_fc_var(in_size, out_size, name):
      # please set a regularizer for the fc weights
      weights = tf.get_variable('weights',
              shape=[in_size, out_size],
              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=5e-2),
              trainable=trainable,
              regularizer=regularizer) # regularizer on the right hand is a global variable

      biases = tf.get_variable('biases',
              shape=[out_size],
              initializer=tf.constant_initializer(value=0.0),
              trainable=trainable,
              regularizer=None)

      return weights, biases
```

One can find that there is a extra `regularizer` for the `weights`, which is used to prevent the values of weights being too large. In CNN, it is important to import such regularizers for the parameters in some ocassions, since values of parameters can be easily uncontrollable<br> 

Untill now, we need to calculate the loss and accuracy of the model.<br>
```python
  softmax_l_15 = tf.nn.softmax(fc_l_14, name="layer15")
  accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=tf.argmax(softmax_l_15, 1))

  loss = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=labels, logits=fc_l_14)
  loss = tf.reduce_mean(loss)

  reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  # the reg_loss will be empty if no regularizer added
  if len(reg_loss) > 0:
      reg_loss = tf.add_n(reg_loss)
      tf.summary.scalar('reg_loss', reg_loss)
      total_loss = loss + reg_loss
  else:
      total_loss = loss
```

The above codes are just to help you understand how to build a model with your own structure. For training and evaluation, see `model_train.py` and `model_eval.py`, where you will see how to save and restore a model to or from a path, how to do summary and track the values of variables.<br>

There is an example of training and evaluation
```bash
  python model_train.py \
    --train_dir='/mymodel/model' \
    --data_dir='/mymodel/cifar10-data'
```

```bash
  python model_eval.py \
    --train_dir='/mymodel/model' \
    --data_dir='/mymodel/cifar10-data'
```
* Please at least understand the following code:
  * mymodel.py
  * model_train.py 
  * model_eval.py

*Note:* The performance of this model may depend on the initialization, sometimes the accuracy gets stuck in 0.1 and does not increase. For this case, just kill it and rerun. The solution of this probelm may be introducing other tricks, like dropout, loss average and batch normalization, or modifying the regularization of weights. Have a try.

Assuming that you have trained the model and save the log file to /mymodel/model/log, for visualization, use
```bash
tensorboard --logdir='/mymodel/model/log'
```
then open the link created in your browser.<br>
The results of tensorboard are like this<br>
![](https://github.com/SuZhuo/DeepLearning-course/raw/master/assignment4/images/graph.jpg)
![](https://github.com/suzhuo/DeepLearning-course/raw/master/assignment4/images/scale1.jpg)
![](https://github.com/suzhuo/DeepLearning-course/raw/master/assignment4/images/scale2.jpg)
![](https://github.com/suzhuo/DeepLearning-course/raw/master/assignment4/images/histogram.jpg)

Now, bonus:<br>
Can you modify the `model_train.py` so that it can load the saved checkpoint and continue to train?<br>

Any problem: email zhuo.su@oulu.fi
