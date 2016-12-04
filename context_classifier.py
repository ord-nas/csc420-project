import tensorflow as tf
import numpy as np
import os
import math
import tensorflow.contrib.slim as slim

import classifier_utils as utils

def compute_probability_neighbourhoods(sess, data, patch_model, bin_size=27, context_length=7):
    # context_length must be odd
    assert context_length % 2 == 1
    
    # First we want to compute everybody's probabilities
    batch_size = 100
    (N, C) = data['labels'].shape
    probabilities = np.zeros((N,C)) 
    for i in xrange(0, N, batch_size):
        p_batch = sess.run(patch_model.inference_predictions, feed_dict={
            patch_model.patch_tensor:data['patches'][i:i+batch_size]*255,
        })
        probabilities[i:i+batch_size] = p_batch

    # Next group all of the input patches by image
    patches_by_img = {}
    for i in xrange(N):
        img_id = data['img_ids'][i]
        if img_id not in patches_by_img:
            patches_by_img[img_id] = list()
        patches_by_img[img_id].append(i)

    # Figure out the maximum width and height we need to care about
    imgW = np.max(data['centres'][:, 0])
    imgH = np.max(data['centres'][:, 1])
    # Convert that to bin indices
    bin_count_x = int(math.floor(imgW / bin_size)) + 1
    bin_count_y = int(math.floor(imgH / bin_size)) + 1

    # Create an array to keep track of each patch's neighbourhood of
    # probability weight.
    weight_neighbourhoods = np.zeros((N, context_length, context_length, C))

    # Now consider each image
    for (img_id, indices) in patches_by_img.iteritems():
        # Build a 2D map of probability weight over this entire image
        weight = np.zeros((bin_count_y, bin_count_x, C))
        for i in indices:
            (x, y) = data['centres'][i]
            bin_x = int(math.floor(x / bin_size))
            bin_y = int(math.floor(y / bin_size))
            weight[bin_y, bin_x] += probabilities[i]

        # Once we've built the map, let's read off the neighbourhoods
        # and insert them into our big array.
        delta = (context_length-1)/2
        weight = np.pad(weight, ((delta, delta), (delta, delta), (0, 0)), 'constant')
        for i in indices:
            (x, y) = data['centres'][i]
            bin_x = int(math.floor(x / bin_size)) + delta
            bin_y = int(math.floor(y / bin_size)) + delta
            weight_neighbourhoods[i] = weight[bin_y-delta:bin_y+delta+1,
                                              bin_x-delta:bin_x+delta+1]

    return (probabilities, weight_neighbourhoods)

# Builder functions

def classifier_model(probability_batch, neighbourhood_batch):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        biases_regularizer=None):
        with slim.arg_scope([slim.dropout], keep_prob=0.5):
            # Combine the probabilities and the neighbourhood together
            # as a single flat vector
            net = tf.concat(1,
                            [probability_batch,
                             slim.flatten(neighbourhood_batch, scope='1_flatten')],
                            name='1_concat')
            net = slim.fully_connected(net, 512, scope='1_fc')
            net = slim.dropout(net, scope='1_dropout')
            net = slim.fully_connected(net, 512, scope='2_fc')
            net = slim.dropout(net, scope='1_dropout')
            net = slim.fully_connected(net, 512, scope='3_fc')
            net = slim.dropout(net, scope='1_dropout')
            net = slim.fully_connected(net, 4, activation_fn=None, scope='4_fc')
            return net

def get_training_op(prediction_logits, labels, learning_rate=5e-4):
    loss = slim.losses.softmax_cross_entropy(prediction_logits, labels)
    total_loss = slim.losses.get_total_loss()
    optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate)
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    return (train_op, loss)

# Main model class
class ContextModel():
    def __init__(self, learning_rate=5e-4, context_length=7):
        self.probability_tensor = tf.placeholder(dtype='float32', shape=(None, 4))
        self.neighbourhood_tensor = tf.placeholder(
            dtype='float32', shape=(None, context_length, context_length, 4))
        self.label_tensor = tf.placeholder(dtype='int32', shape=(None, 4))
        with tf.variable_scope("context_classifier"):
            with slim.arg_scope([slim.dropout], is_training=True):
                self.prediction_logits = classifier_model(
                    self.probability_tensor, self.neighbourhood_tensor)
        with tf.variable_scope("context_classifier", reuse=True):
            with slim.arg_scope([slim.dropout], is_training=False):
                self.inference_prediction_logits = classifier_model(
                    self.probability_tensor, self.neighbourhood_tensor)
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
                   train_probabilities,
                   train_neighbourhoods,
                   train_labels,
                   test_probabilities,
                   test_neighbourhoods,
                   test_labels,
                   epochs,
                   batch_size,
                   reset=True):
        tr_loss = []
        tst_loss = []
        N = train_probabilities.shape[0]
        assert train_neighbourhoods.shape[0] == N
        assert train_labels.shape[0] == N
        if reset:
            sess.run(tf.initialize_all_variables())
        for e in xrange(epochs):
            for i in xrange(0, N, batch_size):
                [_, loss, p] = sess.run([self.train_op, self.loss, self.predictions], feed_dict={
                        self.probability_tensor:train_probabilities[i:i+batch_size],
                        self.neighbourhood_tensor:train_neighbourhoods[i:i+batch_size],
                        self.label_tensor:train_labels[i:i+batch_size],
                    })
                step = i / batch_size
                if step % 25 == 0:
                    [test_loss, dacc, acc, f] = sess.run(
                        [self.loss, self.dropout_accuracy, self.accuracy, self.f1], feed_dict={
                        self.probability_tensor:test_probabilities,
                        self.neighbourhood_tensor:test_neighbourhoods,
                        self.label_tensor:test_labels,
                    })
                    print "Epoch %d, step %d, training loss %f, test_loss %f, accuracy = %f/%f, f1 = %f" % (e, step, loss, test_loss, dacc, acc, f)
                    tr_loss.append(loss)
                    tst_loss.append(test_loss)
            [test_loss, dacc, acc, f, conf] = sess.run(
                [self.loss, self.dropout_accuracy, self.accuracy, self.f1, self.confusion], feed_dict={
                self.probability_tensor:test_probabilities,
                self.neighbourhood_tensor:test_neighbourhoods,
                self.label_tensor:test_labels,
            })
            print "End of epoch %d, training loss %f, test_loss %f, accuracy = %f/%f, f1 = %f" % (e, loss, test_loss, dacc, acc, f)
            print "Confusion matrix:"
            print conf
        return (tr_loss, tst_loss)

