import tensorflow as tf
import numpy as np
import os
import math
import tensorflow.contrib.slim as slim

import classifier_utils as utils

# Builder functions
def classifier_model(image_batch):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        biases_initializer=tf.zeros_initializer,
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        biases_regularizer=None):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='VALID'):
            with slim.arg_scope([slim.dropout], keep_prob=0.8):
                net = image_batch
                net = slim.conv2d(net, 36, [4, 4], scope='1_conv')
                net = slim.max_pool2d(net, [2, 2], scope='2_max_pool')
                net = slim.conv2d(net, 48, [3, 3], scope='3_conv')
                net = slim.max_pool2d(net, [2, 2], scope='4_max_pool')
                net = slim.flatten(net, scope='4_flatten')
                net = slim.fully_connected(net, 512, scope='5_fc')
                net = slim.dropout(net, scope='5_dropout')
                net = slim.fully_connected(net, 512, scope='6_fc')
                net = slim.dropout(net, scope='6_dropout')
                net = slim.fully_connected(net, 4, activation_fn=None, scope='7_fc')
                return net

def get_training_op(prediction_logits, labels, learning_rate=5e-4):
    loss = slim.losses.softmax_cross_entropy(prediction_logits, labels)
    total_loss = slim.losses.get_total_loss()
    optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate)
    #optimizer = tf.train.AdamOptimizer()
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    return (train_op, loss)

# Main model class
class SoftmaxCNNModel():
    def __init__(self, learning_rate=5e-4):
        self.patch_tensor = tf.placeholder(dtype='float32', shape=(None, 27, 27, 3))
        self.label_tensor = tf.placeholder(dtype='int32', shape=(None, 4))
        with tf.variable_scope("classifier"):
            with slim.arg_scope([slim.dropout], is_training=True):
                self.prediction_logits = classifier_model(self.patch_tensor)
        with tf.variable_scope("classifier", reuse=True):
            with slim.arg_scope([slim.dropout], is_training=False):
                self.inference_prediction_logits = classifier_model(self.patch_tensor)
        (self.train_op, self.loss) = get_training_op(self.prediction_logits,
                                                     self.label_tensor,
                                                     learning_rate=learning_rate)
        self.predictions = slim.softmax(self.prediction_logits)
        self.inference_predictions = slim.softmax(self.inference_prediction_logits)
        self.dropout_accuracy = utils.get_accuracy(self.predictions, self.label_tensor)
        self.accuracy = utils.get_accuracy(self.inference_predictions, self.label_tensor)
        self.f1 = utils.get_weighted_f1(self.inference_predictions, self.label_tensor)
        self.confusion = utils.get_confusion(self.inference_predictions, self.label_tensor)

    def train_loop(self,
                   sess,
                   train_patches,
                   train_labels,
                   test_patches,
                   test_labels,
                   epochs,
                   batch_size,
                   reset=True):
        # Some aliases that make things easier to access
        patch_tensor = self.patch_tensor
        label_tensor = self.label_tensor
        train_op = self.train_op
        softmax_loss = self.loss
        predictions = self.predictions
        dropout_accuracy = self.dropout_accuracy
        accuracy = self.accuracy
        f1 = self.f1
        confusion = self.confusion

        tr_loss = []
        tst_loss = []
        N = train_patches.shape[0]
        assert train_labels.shape[0] == N
        if reset:
            sess.run(tf.initialize_all_variables())
        for e in xrange(epochs):
            for i in xrange(0, N, batch_size):
                [_, loss, p] = sess.run([train_op, softmax_loss, predictions], feed_dict={
                        patch_tensor:train_patches[i:i+batch_size],
                        label_tensor:train_labels[i:i+batch_size],
                    })
                step = i / batch_size
                if step % 25 == 0:
                    [test_loss, dacc, acc, f] = sess.run(
                        [softmax_loss, dropout_accuracy, accuracy, f1], feed_dict={
                        patch_tensor:test_patches,
                        label_tensor:test_labels,
                    })
                    print "Epoch %d, step %d, training loss %f, test_loss %f, accuracy = %f/%f, f1 = %f" % (e, step, loss, test_loss, dacc, acc, f)
                    tr_loss.append(loss)
                    tst_loss.append(test_loss)
            [test_loss, dacc, acc, f, conf] = sess.run(
                [softmax_loss, dropout_accuracy, accuracy, f1, confusion], feed_dict={
                patch_tensor:test_patches,
                label_tensor:test_labels,
            })
            print "End of epoch %d, training loss %f, test_loss %f, accuracy = %f/%f, f1 = %f" % (e, loss, test_loss, dacc, acc, f)
            print "Confusion matrix:"
            print conf
        return (tr_loss, tst_loss)
