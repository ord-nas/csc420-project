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
