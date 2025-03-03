# This file implements a bunch of utilities that are generally useful
# across the classifiers. This includes dataset reading, computing
# interesting statistics, and visualization.

import tensorflow as tf
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import math

# DATASET READING

# Read classifier data for a single image
def get_image_data(filename, categories):
    img = mpimg.imread(filename)
    (name, ext) = os.path.splitext(filename)
    n = len(categories)
    centres = []
    labels = []
    for (i, c) in enumerate(categories):
        d = loadmat(name + ('_%s.mat' % c))['detection']
        centres.append(d)
        one_hot_row = np.zeros((n,), dtype='int')
        one_hot_row[i] = 1
        labels.append(np.tile(one_hot_row, (d.shape[0], 1)))
    centres = np.vstack(centres)
    labels = np.vstack(labels)
    return (img, centres, labels)

# Read all image files in a dataset
def get_dataset(num_images, categories, prefix='Dataset/CRCHistoPhenotypes_2016_04_28/Classification/'):
    all_img = []
    all_centres = []
    all_labels = []
    for i in range(1,num_images+1):
        path = os.path.join(prefix, 'img%d/img%d.bmp' % (i, i))
        (img, centres, labels) = get_image_data(path, categories)
        all_img.append(img)
        all_centres.append(centres)
        all_labels.append(labels)
    return (all_img, all_centres, all_labels)

# Takes the data read from a dataset and constructs individual classification
# examples (i.e. by cropping out little patches around each cell and labeling it
# correctly).
def get_examples(all_img, all_centres, all_labels, H, W):
    patches = []
    output_labels = []
    output_centres = []
    output_img_ids = []
    cnt_dropped = 0
    assert H%2==1
    assert W%2==1
    dx = (W-1)/2
    dy = (H-1)/2
    for (i, (img, centres, labels)) in enumerate(zip(all_img, all_centres, all_labels)):
        (imgH, imgW, _) = img.shape
        for ((x, y), label) in zip(centres, labels):
            x = int(x)
            y = int(y)
            if (x - dx < 0 or
                x + dx >= imgW or
                y - dy < 0 or
                y + dy >= imgH):
                cnt_dropped += 1
                continue
            patches.append(img[y-dy:y+dy+1, x-dx:x+dx+1, :])
            output_labels.append(label)
            output_centres.append([x, y])
            output_img_ids.append(i)
    patches = np.stack(patches)
    output_labels = np.stack(output_labels)
    output_centres = np.array(output_centres)
    output_img_ids = np.array(output_img_ids)
    print "Dropped %d patches because too close to image border" % cnt_dropped
    return (patches, output_labels, output_centres, output_img_ids)

# Expand training data by performing random augmentations on image
# examples. Also balances the number of examples in each category (i.e. it
# effectively upsamples or downsamples each image category so that each has
# exactly desired_cnt_per_category examples.
def expand_training_data(all_imgs, train_patches, train_labels, train_centres, train_img_ids, desired_cnt_per_category):
    np.random.seed(42) # repeatability
    H = train_patches.shape[1]
    W = train_patches.shape[2]
    C = train_labels.shape[1]
    patches = np.zeros(shape=(C*desired_cnt_per_category, H, W, 3), dtype='uint8')
    labels = np.zeros(shape=(C*desired_cnt_per_category, C), dtype='int')
    centres = np.zeros(shape=(C*desired_cnt_per_category, 2))
    img_ids = np.zeros(shape=(C*desired_cnt_per_category,), dtype='int')
    deltas = np.zeros(shape=(C*desired_cnt_per_category, 2), dtype='int') # [dx, dy]
    flips = np.zeros(shape=(C*desired_cnt_per_category,), dtype='bool') # (False=none, True=ud)
    rots = np.zeros(shape=(C*desired_cnt_per_category,), dtype='int') # (0, 90, 180, or 270)
    hsv_factors = np.zeros(shape=(C*desired_cnt_per_category, 3)) # [h, s, v]
    for c in xrange(C):
        # Find examples matching category
        lookup = (train_labels[:,c] == 1)
        my_train_centres = train_centres[lookup, :]
        my_train_img_ids = train_img_ids[lookup]
        cnt = my_train_centres.shape[0]
        assert cnt != 0

        # Initalize some stuff
        for i in xrange(desired_cnt_per_category):
            centre = my_train_centres[i % cnt]
            img_id = my_train_img_ids[i % cnt]
            img = all_imgs[img_id]
            (delta, flip, rot, hsv_factor) = choose_aug(img, centre, H, W)
            patch = apply_aug(img, centre, delta, flip, rot, hsv_factor, H, W)

            offset = c*desired_cnt_per_category + i
            patches[offset] = patch
            labels[offset,c] = 1
            centres[offset] = centre
            img_ids[offset] = img_id
            deltas[offset] = delta
            flips[offset] = flip
            rots[offset] = rot
            hsv_factors[offset] = hsv_factor
    return {
        'patches' : patches,
        'labels' : labels,
        'centres' : centres,
        'img_ids' : img_ids,
        'deltas' : deltas,
        'flips' : flips,
        'rots' : rots,
        'hsv_factors' : hsv_factors,
    }

# Helper function for augmentation; randomly chooses and augmentation to
# perform.
def choose_aug(img, centre, H, W):
    #global how_many_times
    # Do a bunch of work to figure out which deltas are allowed at this position
    assert H%2==1
    assert W%2==1
    halfH = (W-1)/2
    halfW = (H-1)/2
    (imgH, imgW, _) = img.shape
    (x, y) = centre

    # Helper function to check for out of bounds
    def inbounds(dx, dy):
        return (x + dx - halfW >= 0 and
                x + dx + halfW < imgW and
                y + dy - halfH >= 0 and
                y + dy + halfH < imgH)

    # Iterate over each neighbour position
    deltas = []
    d = 3 # Parameter, should ultimately be factored out
    for dx in range(-d, d+1):
        for dy in range(-d, d+1):
            if dx**2 + dy**2 <= d**2 and inbounds(dx, dy):
                deltas.append(np.array([dx, dy]))

    # Choose stuff
    delta = deltas[np.random.choice(len(deltas))]
    #if how_many_times < 10:
    #    print deltas
    #    print delta
    #    how_many_times += 1
    flip = np.random.choice([True, False])
    rot = np.random.choice([0, 90, 180, 270])
    h = np.random.uniform(0.95, 1.05) # Paramters, should ultimately be factored out
    s = np.random.uniform(0.9, 1.1) # Paramters, should ultimately be factored out
    v = np.random.uniform(0.9, 1.1) # Paramters, should ultimately be factored out
    hsv = np.array([h, s, v])
    return (delta, flip, rot, hsv)

# Helper function for augmentation; actually performs the given transformation.
def apply_aug(img, centre, delta, flip, rot, hsv_factors, H, W):
    assert H%2==1
    assert W%2==1
    halfH = (W-1)/2
    halfW = (H-1)/2

    (x, y) = centre
    (dx, dy) = delta
    patch = img[y+dy-halfH:y+dy+halfW+1,x+dx-halfW:x+dx+halfW+1,:]
    if flip:
        patch = np.flipud(patch)
    patch = np.rot90(patch, rot / 90)
    patch = matplotlib.colors.rgb_to_hsv(patch / 255.0)
    patch = np.maximum(0.0, np.minimum(1.0, patch * hsv_factors))
    patch = np.maximum(0, np.minimum(255, np.round(255 * matplotlib.colors.hsv_to_rgb(patch)))).astype('uint8')
    return patch

# Read in the dataset, split into train a test set, and perform
# augmentation and shuffling. Returns (train, test) pair, where each
# is a dictionary of data.
def get_augmented_dataset(
        categories,
        desired_cnt_per_category=15000,
        num=100,
        H=27,
        W=27,
        train_fraction=0.8):
    # Read in the raw data
    (raw_imgs, raw_centres, raw_labels) = get_dataset(num, categories)

    # Extract example patches from the data
    (patches, labels, centres, img_ids) = get_examples(
        raw_imgs, raw_centres, raw_labels, H, W)

    # Organize examples into training and test data
    N = patches.shape[0]
    num_train = int(train_fraction * N)
    np.random.seed(0) # predictable shuffling for now
    perm = np.random.permutation(N)
    train_patches, test_patches = np.split(patches[perm], [num_train])
    train_labels, test_labels = np.split(labels[perm], [num_train])
    train_centres, test_centres = np.split(centres[perm], [num_train])
    train_img_ids, test_img_ids = np.split(img_ids[perm], [num_train])
    # Convert to float
    train_patches = train_patches / 255.0
    test_patches = test_patches / 255.0

    # Expand/augment the training data
    sorted_train_dict = expand_training_data(
        raw_imgs, train_patches, train_labels, train_centres, train_img_ids,
        desired_cnt_per_category)
    # Convert to float
    sorted_train_dict['patches'] = sorted_train_dict['patches'] / 255.0

    # Suffle the augmented training data
    trainN = sorted_train_dict['patches'].shape[0]
    np.random.seed(123) # predictable shuffling for now
    perm = np.random.permutation(trainN)
    train_dict = {k : v[perm] for (k, v) in sorted_train_dict.iteritems()}

    # Construct the test data dictionary
    test_dict = {
        'patches' : test_patches,
        'labels' : test_labels,
        'centres' : test_centres,
        'img_ids' : test_img_ids,
    }

    return train_dict, test_dict

# Same as above, but we want to divide the dataset into train and test
# *by image* instead of by patch as before. This means that certain
# images will be entirely training data, including all their patches,
# and others will be entirely test data.
def get_augmented_dataset_divided_per_image(
        categories,
        desired_cnt_per_category=15000,
        num=100,
        H=27,
        W=27,
        train_fraction=0.8):
    # Read in the raw data
    (raw_imgs, raw_centres, raw_labels) = [np.array(x) for x in get_dataset(num, categories)]

    np.random.seed(9999) # predictable shuffling for now

    # Organize examples into training and test data
    N = len(raw_imgs)
    num_train = int(train_fraction * N)
    np.random.seed(0) # predictable shuffling for now
    perm = np.random.permutation(N)
    train_raw_imgs, test_raw_imgs = np.split(raw_imgs[perm], [num_train])
    train_raw_labels, test_raw_labels = np.split(raw_labels[perm], [num_train])
    train_raw_centres, test_raw_centres = np.split(raw_centres[perm], [num_train])
    shuffled_img_ids = np.arange(N)[perm]
    
    # Extract patches from both the train and test images
    (train_patches, train_labels, train_centres, train_img_ids) = get_examples(
        train_raw_imgs, train_raw_centres, train_raw_labels, H, W)
    train_img_ids = np.array([shuffled_img_ids[i] for i in train_img_ids])
    (test_patches, test_labels, test_centres, test_img_ids) = get_examples(
        test_raw_imgs, test_raw_centres, test_raw_labels, H, W)
    test_img_ids = np.array([shuffled_img_ids[i+num_train] for i in test_img_ids])

    # Convert to float
    train_patches = train_patches / 255.0
    test_patches = test_patches / 255.0

    # Expand/augment the training data
    sorted_train_dict = expand_training_data(
        raw_imgs, train_patches, train_labels, train_centres, train_img_ids,
        desired_cnt_per_category)
    # Convert to float
    sorted_train_dict['patches'] = sorted_train_dict['patches'] / 255.0

    # Suffle the augmented training data
    trainN = sorted_train_dict['patches'].shape[0]
    perm = np.random.permutation(trainN)
    train_dict = {k : v[perm] for (k, v) in sorted_train_dict.iteritems()}

    # Construct the test data dictionary
    test_dict = {
        'patches' : test_patches,
        'labels' : test_labels,
        'centres' : test_centres,
        'img_ids' : test_img_ids,
    }

    return train_dict, test_dict

# Same as above, but skips traning data augmentation
def get_dataset_divided_per_image(
        categories,
        desired_cnt_per_category=15000,
        num=100,
        H=27,
        W=27,
        train_fraction=0.8):
    # Read in the raw data
    (raw_imgs, raw_centres, raw_labels) = [np.array(x) for x in get_dataset(num, categories)]

    # Okay, so we want to divide the dataset into train and test *by
    # image* instead of by patch as before. This means that certain
    # images will be entirely training data, including all their
    # patches, and others will be entirely test data.
    np.random.seed(9999) # predictable shuffling for now

    # Organize examples into training and test data
    N = len(raw_imgs)
    num_train = int(train_fraction * N)
    np.random.seed(0) # predictable shuffling for now
    perm = np.random.permutation(N)
    train_raw_imgs, test_raw_imgs = np.split(raw_imgs[perm], [num_train])
    train_raw_labels, test_raw_labels = np.split(raw_labels[perm], [num_train])
    train_raw_centres, test_raw_centres = np.split(raw_centres[perm], [num_train])
    shuffled_img_ids = np.arange(N)[perm]
    
    # Extract patches from both the train and test images
    (train_patches, train_labels, train_centres, train_img_ids) = get_examples(
        train_raw_imgs, train_raw_centres, train_raw_labels, H, W)
    train_img_ids = np.array([shuffled_img_ids[i] for i in train_img_ids])
    (test_patches, test_labels, test_centres, test_img_ids) = get_examples(
        test_raw_imgs, test_raw_centres, test_raw_labels, H, W)
    test_img_ids = np.array([shuffled_img_ids[i+num_train] for i in test_img_ids])

    # Convert to float
    train_patches = train_patches / 255.0
    test_patches = test_patches / 255.0

    # Construct the train data dictionary
    sorted_train_dict = {
        'patches' : train_patches,
        'labels' : train_labels,
        'centres' : train_centres,
        'img_ids' : train_img_ids,
    }

    # Suffle the training data
    trainN = sorted_train_dict['patches'].shape[0]
    perm = np.random.permutation(trainN)
    train_dict = {k : v[perm] for (k, v) in sorted_train_dict.iteritems()}

    # Construct the test data dictionary
    test_dict = {
        'patches' : test_patches,
        'labels' : test_labels,
        'centres' : test_centres,
        'img_ids' : test_img_ids,
    }

    return train_dict, test_dict


# CLASSIFICATION STATISTICS

# Simple % of examples correctly classified.
def get_accuracy(predictions, labels):
    p = tf.argmax(predictions, dimension=1)
    l = tf.argmax(labels, dimension=1)
    correct = tf.equal(p, l)
    #print correct.get_shape()
    return tf.reduce_mean(tf.cast(correct, tf.float32))

# F1-score is harmonic mean of precision and recall, see:
# https://en.wikipedia.org/wiki/F1_score. Weighted F1-score is a weighted
# average of the per-category f1 scores (which are computed in a one-vs-all kind
# of way). The contribution of each category is weighted by the % of the labels
# which have that category.
def get_weighted_f1(predictions, labels):
    p = tf.argmax(predictions, dimension=1)
    l = tf.argmax(labels, dimension=1)
    f1s = []
    totals = []
    C = labels.get_shape()[1]
    def cnt(t):
        return tf.reduce_sum(tf.cast(t, tf.float32))
    def w(c, msg, t):
        return tf.Print(t, [t], message=(msg + "_" + str(c) + ":"))
        #return t
    for c in xrange(C):
        c = np.array([c], dtype='int64')
        true_positives = w(c, "true_positive", cnt(tf.logical_and(tf.equal(c, p), tf.equal(c, l))))
        #false_positives = w(c, "false_positive", cnt(tf.logical_and(tf.equal(c, p), tf.not_equal(c, l))))
        positive_labels = w(c, "positive_labels", cnt(tf.equal(c, l)))
        positive_predictions = w(c, "positive_predictions", cnt(tf.equal(c, p)))
        recall = w(c, "recall", tf.div(true_positives, positive_labels))
        precision = w(c, "precision", tf.div(true_positives, positive_predictions))
        f1 = w(c, "f1", tf.div(2*tf.mul(precision, recall),
                               tf.add(precision, recall)))
        f1s.append(f1)
        totals.append(positive_labels)
    weights = w(0, "weights", tf.div(tf.pack(totals), tf.reduce_sum(tf.pack(totals))))
    f1_avg = w(0, "f1_avg", tf.reduce_sum(tf.mul(weights, tf.pack(f1s))))
    return f1_avg

# Confusion matrix, https://en.wikipedia.org/wiki/Confusion_matrix
def get_confusion(predictions, labels):
    C = labels.get_shape()[1]
    p_hot = tf.transpose(tf.one_hot(tf.argmax(predictions, dimension=1), depth=C, dtype=tf.int32))
    l_hot = tf.transpose(tf.one_hot(tf.argmax(labels, dimension=1), depth=C, dtype=tf.int32)) # not really necessary, other than the transpose
    # Results is CxCxbatch_size, and results[i,j,k] is one iff for example k, the model predicted j and the right answer is i
    results = tf.mul(p_hot, l_hot[:,tf.newaxis,:])
    # Sum along the batch dimension to get the confusion matrix
    return tf.reduce_sum(results, reduction_indices=2)

# VISUALIZATION

# Label an image with differently-coloured points representing the cell
# classifications.
def draw(img, centres, labels, categories, figsize=(10,10)):
    colours = ['red', 'blue', 'green', 'yellow', 'white']
    plt.figure(figsize=figsize)
    plt.imshow(img)
    for (i, category) in enumerate(categories):
        x = centres[labels[:,i] == 1, 0]
        y = centres[labels[:,i] == 1, 1]
        colour = colours[i % len(colours)]
        plt.scatter(x, y, c=colour, label=category)
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), scatterpoints=1)
