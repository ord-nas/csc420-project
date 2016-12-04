import numpy as np

# gradient of y wrt w given x, activations, and C
# There are M training examples and N features (including bias)
# x is MxN
# y is Mx1
# w is Nx1
# activations is Mx1
# C is scalar
# Returns Nx1 column of gradients
def get_gradient(x, y, w, activations, C):
    values = -1 * x * y # MxN
    mask = (activations[:, 0] == 0.0)
    values[mask, :] = 0
    avg = np.mean(values, axis=0, keepdims=True)
    return C*avg.T + w

# x is MxN
# y is Mx1
# w is Nx1
# Returns Mx1 column of activations
def get_activations(x, y, w):
    #print "x", x.shape
    #print "y", y.shape
    #print "w", w.shape
    return np.maximum(0, 1 - (y * np.matmul(x, w)))

# w is Nx1
# activations is Mx1
# C is scalar
# Returns a scalar
def get_cost(w, activations, C):
    regularizer = 0.5 * np.sum(np.square(w))
    return C * np.mean(activations) + regularizer

# x is MxN
# y is Mx1
# w is Nx1
# C is scalar
# Returns scalar cost and Nx1 column of gradients
def get_cost_and_gradient(x, y, w, C):
    activations = get_activations(x, y, w)
    cost = get_cost(w, activations, C)
    gradient = get_gradient(x, y, w, activations, C)
    return (cost, gradient)

# x is MxN
# y is Mx1
# w is Nx1
# C is scalar
# Returns current cost, new w
def training_step(x, y, w, C, learning_rate):
    (cost, gradient) = get_cost_and_gradient(x, y, w, C)
    step = -1 * learning_rate * gradient
    return (cost, w + step)

# x is MxN
# ws is NxL (if there are L classes)
# Returns a tuple:
# 1) array of size M, where each entry is index of predicted class
# 2) array of MxL, where each entry is the per-category SVM prediction
def get_multiclass_predictions(x, ws, add_ones=True):
    # Add 1s to x for bias term
    M = x.shape[0]
    if add_ones:
        x = np.append(x, np.ones((M, 1)), axis=1)
    values = np.matmul(x, ws)
    predictions = np.argmax(values, axis=1)
    per_class_predictions = np.where(values > 0, 1, -1)
    return (predictions, per_class_predictions)

def train_loop(x, y, test_x, test_y, C, learning_rate, batch_size, epochs):
    pass

def get_np_weighted_f1(predictions, actual, num_labels):
    try:
        f1s = []
        totals = []
        for label in xrange(num_labels):
            #print "Label", label
            true_positives = np.count_nonzero(np.logical_and(np.equal(label, actual),
                                                             np.equal(label, predictions)))
            #print "TP", true_positives
            positive_labels = np.count_nonzero(np.equal(label, actual))
            #print "PL", positive_labels
            positive_predictions = np.count_nonzero(np.equal(label, predictions))
            #print "PP", positive_predictions
            recall = float(true_positives) / float(positive_labels)
            #print "Recall", recall
            precision = float(true_positives) / float(positive_predictions)
            #print "Precision", precision
            f1 = 2*precision*recall / (precision + recall)
            #print "F1", f1
            f1s.append(f1)
            totals.append(positive_labels)
        total_weight = float(sum(totals))
        #print "Total weight", total_weight
        weights = [w / total_weight for w in totals]
        #print "Weights", weights
        f1_avg = sum([w * f1 for (w, f1) in zip(weights, f1s)])
        #print "F1_avg", f1_avg
        return f1_avg
    except ZeroDivisionError:
        return 0

def status_report(e, step, x, y, val_x, val_y, test_x, test_y, ws, C, quiet=False):
    training_losses = []
    validation_losses = []
    test_losses = []
    for label in xrange(len(ws)):
        w = ws[label]
        (train_loss, _) = get_cost_and_gradient(x, y[:, label:label+1], w, C)
        (val_loss, _) = get_cost_and_gradient(val_x, val_y[:, label:label+1], w, C)
        (test_loss, _) = get_cost_and_gradient(test_x, test_y[:, label:label+1], w, C)
        training_losses.append(train_loss)
        validation_losses.append(val_loss)
        test_losses.append(test_loss)
    (predictions, per_class_predictions) = get_multiclass_predictions(
        test_x, np.hstack(ws), add_ones=False)
    # Per-class accuracy
    per_class_accuracy = np.mean(np.equal(test_y, per_class_predictions).astype('float32'),
                                 axis=0)
    # Overall accuracy/f1 (test)
    actual = np.argmax(test_y, axis=1)
    correct = np.equal(predictions, actual)
    overall_test_accuracy = np.mean(correct.astype('float32'))
    test_weighted_f1 = get_np_weighted_f1(predictions, actual, len(ws))
    # Overall accuracy/f1 (validation)
    (predictions, _) = get_multiclass_predictions(val_x, np.hstack(ws), add_ones=False)
    actual = np.argmax(val_y, axis=1)
    correct = np.equal(predictions, actual)
    overall_val_accuracy = np.mean(correct.astype('float32'))
    val_weighted_f1 = get_np_weighted_f1(predictions, actual, len(ws))
    
    if not quiet:
        print ("Epoch %s, step %s:\n\ttrain loss %s,\n\tvalidation loss %s," +
               "\n\ttest loss %s,\n\tper-category test accuracy %s," +
               "\n\toverall validation accuracy %f,\n\toverall test accuracy %f," +
               "\n\tvalidation f1 %f,\n\ttest f1 %f") % (
            e, step, training_losses, validation_losses, test_losses, per_class_accuracy,
            overall_val_accuracy, overall_test_accuracy, val_weighted_f1, test_weighted_f1)
    
    return {
        'train_loss': training_losses,
        'val_loss': validation_losses,
        'test_loss': test_losses,
        'val_accuracy': overall_val_accuracy,
        'test_accuracy': overall_test_accuracy,
        'val_f1': val_weighted_f1,
        'test_f1': test_weighted_f1,
    }

def multiclass_train_loop(x, y, val_x, val_y, test_x, test_y, C, learning_rate, batch_size, epochs,
                          quiet=False, init_ws=None):
    # Make status variables global so that we can kill the training loop at any
    # time if it's taking too long, and as long as we keep the shell alive, we
    # will be able to inspect our current progress & results.
    global loss, val_loss, test_loss, params # should be accessed as variable[timestep][category]
    loss = []
    val_loss = []
    test_loss = []
    params = []
    
    global val_accuracy, test_accuracy, val_f1, test_f1 # should be accesses as variable[timestep]
    val_accuracy = []
    test_accuracy = []
    val_f1 = []
    test_f1 = []
    
    np.random.seed(31415) # repeatability

    # Sanity-check the input sizes
    L = y.shape[1]
    (M, N) = x.shape
    assert y.shape == (M, L)
    (M_val, N_val) = val_x.shape
    assert N_val == N
    assert val_y.shape == (M_val, L)
    (M_test, N_test) = test_x.shape
    assert N_test == N
    assert test_y.shape == (M_test, L)
    
    # Convert from one-hot labels to per-class -1/+1 labels
    y = y * 2 - 1
    val_y = val_y * 2 - 1
    test_y = test_y * 2 - 1

    # Intialize weights for each class
    ws = [np.random.normal(size=(N+1, 1)) for _ in xrange(L)] if init_ws is None \
         else np.hsplit(init_ws, init_ws.shape[1])
    # Add 1s to x for bias term
    x = np.append(x, np.ones((M, 1)), axis=1)
    test_x = np.append(test_x, np.ones((M_test, 1)), axis=1)
    val_x = np.append(val_x, np.ones((M_val, 1)), axis=1)

    # Run training
    for e in xrange(epochs):
        for i in xrange(0, M, batch_size):
            for label in xrange(L):
                w = ws[label]
                (cost, new_w) = training_step(
                    x[i:i+batch_size], y[i:i+batch_size, label:label+1], w, C, learning_rate)
                ws[label] = new_w
            step = i / batch_size
            if step % 50 == 0:
                status_dict = status_report(
                    e, step, x[i:i+batch_size], y[i:i+batch_size],
                    val_x, val_y, test_x, test_y, ws, C, quiet=quiet)
                loss.append(status_dict['train_loss'])
                val_loss.append(status_dict['val_loss'])
                test_loss.append(status_dict['test_loss'])
                val_accuracy.append(status_dict['val_accuracy'])
                test_accuracy.append(status_dict['test_accuracy'])
                val_f1.append(status_dict['val_f1'])
                test_f1.append(status_dict['test_f1'])
                params.append(list(ws))
    
    status_report("FINAL", "FINAL", x[0:batch_size], y[0:batch_size],
                  val_x, val_y, test_x, test_y, ws, C, quiet=False)

    # Return per-parameter weights
    return np.hstack(ws)
