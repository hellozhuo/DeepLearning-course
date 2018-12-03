import tensorflow as tf

import numpy as np

class Mymodel:
    """
    A trainable version my own designed model.
    """

    def __init__(self, weight_decay=0.004, dropout=0.5):

        self.dropout = dropout
        self.regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    def build(self, images, labels, train_mode=True):
        self.trainable = train_mode

        print(images.get_shape().as_list())
        assert images.get_shape().as_list()[1:] == [32, 32, 3]

        with tf.name_scope('Block-1'):
            self.conv_l_1 = self.conv_layer(images, 3, 8, "layer1")
            self.conv_l_2 = self.conv_layer(self.conv_l_1, 8, 8, "layer2")
            self.conv_l_3 = self.conv_layer(self.conv_l_2, 8, 8, "layer3")
            self.pool_l_4 = self.max_pool(self.conv_l_3, 'layer4')

        with tf.name_scope('Block-2'):
            self.conv_l_5 = self.conv_layer(self.pool_l_4, 8, 64, "layer5")
            self.conv_l_6 = self.conv_layer(self.conv_l_5, 64, 64, "layer6")
            self.conv_l_7 = self.conv_layer(self.conv_l_6, 64, 64, "layer7")
            self.pool_l_8 = self.max_pool(self.conv_l_7, 'layer8')

        with tf.name_scope('Block-3'):
            self.conv_l_9 = self.conv_layer(self.pool_l_8, 64, 64, "layer9")
            self.conv_l_10 = self.conv_layer(self.conv_l_9, 64, 64, "layer10")
            self.conv_l_11 = self.conv_layer(self.conv_l_10, 64, 64, "layer11")
            self.pool_l_12 = self.max_pool(self.conv_l_11, 'layer12')

        with tf.name_scope('Block-FC'):
            self.fc_l_13 = self.fc_layer(self.pool_l_12, 1024, "layer13")
            self.fc_l_14 = self.fc_layer(self.fc_l_13, 10, "layer14")
            # self.fc_l_14 = tf.nn.relu(self.fc_l_14)

        with tf.name_scope('Block-OUTPUT'):
            self.softmax_l_15 = tf.nn.softmax(self.fc_l_14, name="layer15")
            self.accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=tf.argmax(self.softmax_l_15, 1))
            tf.summary.scalar('accuracy', self.accuracy[0])

            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=self.fc_l_14)
            self.loss = tf.reduce_mean(self.loss)
            tf.summary.scalar('losses', self.loss)
            # self.cost = tf.reduce_sum((self.softmax_l_15 - labels) ** 2)

            self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if len(self.reg_loss) > 0:
                self.reg_loss = tf.add_n(self.reg_loss)
                tf.summary.scalar('reg_loss', self.reg_loss)
                self.total_loss = self.loss + self.reg_loss
            else:
                self.total_loss = self.loss
            tf.summary.scalar('total_loss', self.total_loss)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, out_size, name):
        with tf.variable_scope(name):
            batch_size = bottom.get_shape()[0]
            x = tf.reshape(bottom, [batch_size, -1])
            in_size = x.get_shape()[1]
            weights, biases = self.get_fc_var(in_size, out_size, name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            tf.summary.histogram('weights', weights)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        filters = tf.get_variable('filters', 
                shape=[filter_size, filter_size, in_channels, out_channels],
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=5e-2),
                trainable=self.trainable,
                regularizer=None)
                #regularizer=self.regularizer)

        biases = tf.get_variable('biases',
                shape=[out_channels],
                initializer=tf.constant_initializer(value=0.0),
                trainable=self.trainable,
                regularizer=None)
                #regularizer=self.regularizer)

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        # please set a regularizer for the fc weights
        weights = tf.get_variable('weights',
                shape=[in_size, out_size],
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=5e-2),
                trainable=self.trainable,
                regularizer=self.regularizer)

        biases = tf.get_variable('biases',
                shape=[out_size],
                initializer=tf.constant_initializer(value=0.0),
                trainable=self.trainable,
                regularizer=None)
                #regularizer=self.regularizer)

        return weights, biases

