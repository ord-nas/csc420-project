# This file implements the autoencoder.

import tensorflow as tf
import numpy as np
import os
import math
import tensorflow.contrib.slim as slim

import classifier_utils as utils

# This little function is the definition of the model! Everything else
# is pretty much just infrastructure.
def autoencoder_model(image_batch):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        biases_initializer=tf.zeros_initializer,
                        biases_regularizer=None):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            padding='VALID'):
            with slim.arg_scope([slim.dropout], keep_prob=0.8):
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
                                    normalizer_fn=slim.batch_norm):
                    net = image_batch

                    estack = []
                    def r(x):
                        estack.append(x)
                        return x
                    with tf.variable_scope("encoder"):
                        net = r(slim.conv2d(net, 36, [4, 4], scope='1_conv'))
                        net = r(slim.conv2d(net, 42, [3, 3], scope='2_conv'))
                        net = r(slim.max_pool2d(net, [2, 2], scope='3_max_pool'))
                        net = r(slim.conv2d(net, 48, [4, 4], scope='4_conv'))
                        net = r(slim.max_pool2d(net, [2, 2], scope='5_max_pool'))
                        net = r(slim.flatten(net, scope='5_flatten'))
                        net = r(slim.fully_connected(net, 256, scope='6_fc'))
                        net = r(slim.fully_connected(net, 128, scope='7_fc'))

                    encoded = net

                    dstack = []
                    def r(x):
                        dstack.append(x)
                        return x
                    with tf.variable_scope("decoder"):
                        net = r(slim.fully_connected(net, 256, scope='1_fc'))
                        net = r(slim.fully_connected(net, 4*4*48, scope='2_fc'))
                        net = r(tf.reshape(net, [-1, 4, 4, 48], name='2_reshape'))
                        net = r(upscale(net, [2, 2], name='3_upscale'))
                        net = r(slim.conv2d_transpose(net, 42, [4, 4], scope='4_conv'))
                        net = r(upscale(net, [2, 2], name='5_upscale'))
                        net = r(slim.conv2d_transpose(net, 36, [3, 3], scope='6_conv'))
                        net = r(slim.conv2d_transpose(net, 3, [4, 4], activation_fn=tf.nn.sigmoid, scope='7_conv'))

                    reconstructed = net

                    return (encoded, reconstructed, estack, dstack)

# Helper function to implement an upscaling layer.
def upscale(images, scales, name):
    (yscale, xscale) = scales
    (batch_size, height, width, channels) = images.get_shape().as_list()
    with tf.name_scope(name):
        return tf.image.resize_images(images, height*yscale, width*xscale,
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# Create an L2 loss tensor. Create a training tensor that uses the
# momentum optimizer.
def get_training_op(reconstructed_tensor, patch_tensor, learning_rate):
    difference = reconstructed_tensor - patch_tensor
    square_difference = tf.square(difference)
    mean_squared_error = tf.reduce_mean(square_difference)
    slim.losses.add_loss(mean_squared_error)
    total_loss = slim.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    return (train_op, mean_squared_error)

# Main model class
class AutoencoderModel():
    # On initialization, construct the model and all placeholders.
    def __init__(self, learning_rate, context_length=7):
        self.patch_tensor = tf.placeholder(dtype='float32', shape=(None, 27, 27, 3))
        self.lr_tensor = learning_rate
        with tf.variable_scope("autoencoder"):
            with slim.arg_scope([slim.dropout], is_training=True), \
                 slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                                normalizer_params={'is_training':True}):
                (self.encoded_tensor, self.reconstructed_tensor,
                 self.estack, self.dstack) = autoencoder_model_3(patch_tensor)
        with tf.variable_scope("autoencoder", reuse=True):
            with slim.arg_scope([slim.dropout], is_training=False), \
                 slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                                normalizer_params={'is_training':False}):
                (self.eval_encoded_tensor, self.eval_reconstructed_tensor,
                 self.eval_estack, self.eval_dstack) = autoencoder_model_3(patch_tensor)
        (self.train_op, self.loss_tensor) = get_training_op(self.reconstructed_tensor,
                                                            self.patch_tensor,
                                                            learning_rate)

    # This is the main loop which trains the model.
    def train_loop(self, sess, train_patches, test_patches, epochs, batch_size, reset=True):
        if not os.path.exists("imgs/train"):
            os.makedirs("imgs/train")
        if not os.path.exists("imgs/test"):
            os.makedirs("imgs/test")
        tr_loss = []
        tst_loss = []
        N = train_patches.shape[0]
        if reset:
            sess.run(tf.initialize_all_variables())
        for e in xrange(epochs):
            for i in xrange(0, N, batch_size):
                [total_loss, loss] = sess.run([self.train_op, self.loss_tensor], feed_dict={
                        self.lr_tensor:0.001,
                        self.patch_tensor:train_patches[i:i+batch_size],
                    })
                step = i / batch_size
                if step % 50 == 0:
                    [test_loss] = sess.run([self.loss_tensor], feed_dict={
                        self.patch_tensor:test_patches[:100],
                    })
                    test_loss = 0
                    print "Epoch %d, step %d, total loss %f, training loss %f, test_loss %f" % (e, step, total_loss, loss, test_loss)
                    tr_loss.append(loss)
                    tst_loss.append(test_loss)
                    imgs = sess.run(self.eval_reconstructed_tensor, feed_dict={
                        self.patch_tensor:train_patches[:16],
                    })
                    for i in range(16):
                        plt.subplot(4,4,i+1)
                        plt.imshow(imgs[i])
                    plt.savefig("imgs/train/%d_%d.png" % (e, step))
                    imgs = sess.run(self.eval_reconstructed_tensor, feed_dict={
                        self.patch_tensor:test_patches[:16],
                    })
                    for i in range(16):
                        plt.subplot(4,4,i+1)
                        plt.imshow(imgs[i])
                    plt.savefig("imgs/test/%d_%d.png" % (e, step))
            # End-of-epoch printing
            [test_loss] = sess.run([self.loss_tensor], feed_dict={
                self.patch_tensor:test_patches[:100],
            })
            test_loss = 0
            print "End of epoch %d, training loss %f, test_loss %f" % (e, loss, test_loss)
            # Save the model
            saver = tf.train.Saver()
            os.makedirs("autoencoder_models/autosave/v%d" % e)
            save_path = saver.save(sess, "autoencoder_models/autosave/v%d/model.ckpt" % e)
            print "Saved to:", save_path
        return (tr_loss, tst_loss)
