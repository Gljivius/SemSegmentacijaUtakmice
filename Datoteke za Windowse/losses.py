import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def add_loss_summaries(total_loss):
  """Add summaries for losses in model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])


  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.

  for l in losses + [total_loss]:
    #print(l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))
    #tf.summary.scalar([l.op.name + ' (raw)'], l)
    #tf.summary.scalar([l.op.name], loss_averages.average(l))

  return loss_averages_op


def total_loss_sum(losses):
  # Assemble all of the losses for the current tower only.
  #losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
  #print(losses)
  # Calculate the total loss for the current tower.
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  #print(regularization_losses)
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
  return total_loss


#def cross_entropy_loss(logits, labels, num_labels):
#  print('Loss: Cross Entropy Loss')
#  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
#  one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
#  logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
#  xent_loss = slim.losses.cross_entropy_loss(logits_1d, one_hot_labels)
#  return xent_loss


def cross_entropy_loss(logits, labels, num_labels):
  print('Loss: Cross Entropy Loss')
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  with tf.name_scope(None, 'WeightedCrossEntropyLoss', [logits, labels]):
    labels = tf.reshape(labels, shape=[num_examples])
    num_labels = tf.to_float(tf.reduce_sum(num_labels))
    one_hot_labels = tf.one_hot(tf.to_int64(labels / 30), FLAGS.num_classes, 1, 0)
    one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])
    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
    # TODO
    #log_softmax = tf.log(tf.nn.softmax(logits_1d))
    log_softmax = tf.nn.log_softmax(logits_1d)
    xent = tf.reduce_sum(-tf.multiply(tf.to_float(one_hot_labels), log_softmax), 1)
    #weighted_xent = tf.multiply(weights, xent)
    #weighted_xent = xent

    total_loss = tf.div(tf.reduce_sum(xent), tf.to_float(num_labels), name='value')

    tf.add_to_collection('_losses', total_loss)
    return total_loss

def bayesian_loss(logits, labels, num_labels, apriori):
  print('Loss: Bayesian Cross Entropy Loss')
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  with tf.name_scope(None, 'BayesianCrossEntropyLoss', [logits, labels]):
    labels = tf.reshape(labels, shape=[num_examples])
    num_labels = tf.to_float(tf.reduce_sum(num_labels))
    one_hot_labels = tf.one_hot(tf.to_int64(labels / 30), FLAGS.num_classes, 1, 0)
    one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])
    one_hot_labels = tf.multiply(tf.to_float(one_hot_labels), tf.to_float(apriori))
    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
    # TODO
    #log_softmax = tf.log(tf.nn.softmax(logits_1d))
    log_softmax = tf.nn.log_softmax(logits_1d)
    xent = tf.reduce_sum(-tf.multiply(one_hot_labels, log_softmax), 1)
    #weighted_xent = tf.multiply(weights, xent)
    #weighted_xent = xent

    total_loss = tf.div(tf.reduce_sum(xent), tf.to_float(num_labels), name='value')

    tf.add_to_collection('_losses', total_loss)
    return total_loss

def hinge_loss(logits, labels, weights, num_labels):
  print('Loss: Hinge Loss')
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  with tf.name_scope(None, 'HingeLoss', [logits, labels]):
    one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
    #codes = tf.nn.softmax(logits_1d)
    codes = tf.nn.l2_normalize(logits_1d, 1)
    # works worse
    l2_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.to_float(one_hot_labels) - codes), 1))
    m = 0.2
    #l2_dist = tf.reduce_sum(tf.square(tf.to_float(one_hot_labels) - codes), 1)
    #m = 0.2 ** 2
    #m = 0.1 ** 2
    #m = 0.3 ** 2
    for i in range(num_classes):
      for j in range(num_classes):
        raise ValueError(1)
    hinge_loss = tf.maximum(tf.to_float(0), l2_dist - m)
    total_loss = tf.reduce_sum(tf.multiply(weights, hinge_loss))

    total_loss = tf.div(total_loss, tf.to_float(num_labels), name='value')
    tf.add_to_collection('_losses', total_loss)

    #tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)
    #tf.nn.l2_loss(t, name=None)
    return total_loss

def weighted_hinge_loss(logits, labels, weights, num_labels):
  print('Loss: Weighted Hinge Loss')
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  with tf.name_scope(None, 'WeightedHingeLoss', [logits, labels]):
    weights = tf.reshape(weights, shape=[num_examples])
    labels = tf.reshape(labels, shape=[num_examples])
    num_labels = tf.to_float(tf.reduce_sum(num_labels))
    one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
    one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])
    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
    #codes = tf.nn.softmax(logits_1d)
    codes = tf.nn.l2_normalize(logits_1d, 1)
    # works worse
    #l2_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.to_float(one_hot_labels) - codes), 1))
    m = 0.2
    l2_dist = tf.reduce_sum(tf.square(tf.to_float(one_hot_labels) - codes), 1)
    #m = 0.2 ** 2
    #m = 0.1 ** 2
    #m = 0.3 ** 2
    hinge_loss = tf.maximum(tf.to_float(0), l2_dist - m)
    total_loss = tf.reduce_sum(tf.multiply(weights, hinge_loss))

    total_loss = tf.div(total_loss, tf.to_float(num_labels), name='value')
    tf.add_to_collection('_losses', total_loss)

    #tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)
    #tf.nn.l2_loss(t, name=None)
    return total_loss


def flip_xent_loss_symmetric(logits, labels, weights, num_labels):
  print('Loss: Cross Entropy Loss')
  num_examples = FLAGS.img_height * FLAGS.img_width
  with tf.name_scope(None, 'CrossEntropyLoss', [logits, labels]):
    labels = tf.reshape(labels, shape=[2, num_examples])
    weights = tf.reshape(weights, shape=[2, num_examples])
    num_labels = tf.to_float(tf.reduce_sum(num_labels))
    #num_labels = tf.to_float(num_labels[0])
    logits_flip = logits[1,:,:,:]
    #weights_flip = weights[1,:]

    logits = logits[0,:,:,:]
    weights = weights[0,:]
    labels = labels[0,:]
    one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
    one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])

    #logits_orig, logits_flip = tf.split(0, 2, logits)
    logits_flip = tf.image.flip_left_right(logits_flip)
    #print(logits[].get_shape())
    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
    logits_1d_flip = tf.reshape(logits_flip, [num_examples, FLAGS.num_classes])
    # TODO
    log_softmax = tf.nn.log_softmax(logits_1d)

    #log_softmax_flip = tf.nn.log_softmax(logits_1d_flip)
    softmax_flip = tf.nn.softmax(logits_1d_flip)
    xent = tf.reduce_sum(tf.multiply(tf.to_float(one_hot_labels), log_softmax), 1)
    weighted_xent = tf.multiply(tf.minimum(tf.to_float(100), weights), xent)
    xent_flip = tf.reduce_sum(tf.multiply(softmax_flip, log_softmax), 1)
    #weighted_xent_flip = tf.multiply(tf.minimum(tf.to_float(100), weights), xent_flip)
    #weighted_xent = tf.multiply(weights, xent)
    #weighted_xent = xent

    #total_loss = tf.div(- tf.reduce_sum(weighted_xent_flip),
    #                    num_labels, name='value')
    total_loss = - tf.div(tf.reduce_sum(weighted_xent) + tf.reduce_sum(xent_flip),
                          num_labels, name='value')

    tf.add_to_collection('_losses', total_loss)
    return total_loss

def flip_xent_loss(logits, labels, weights, num_labels):
  print('Loss: Weighted Cross Entropy Loss')
  num_examples = 2 * FLAGS.img_height * FLAGS.img_width
  labels = tf.reshape(labels, shape=[num_examples])
  weights = tf.reshape(weights, shape=[num_examples])
  num_labels = tf.to_float(tf.reduce_sum(num_labels))
  with tf.name_scope(None, 'WeightedCrossEntropyLoss', [logits, labels]):
    one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
    one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])
    #print(logits[].get_shape())
    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
    # TODO
    #log_softmax = tf.log(tf.nn.softmax(logits_1d))
    log_softmax = tf.nn.log_softmax(logits_1d)
    xent = tf.reduce_sum(tf.multiply(tf.to_float(one_hot_labels), log_softmax), 1)
    #weighted_xent = tf.multiply(weights, xent)
    weighted_xent = tf.multiply(tf.minimum(tf.to_float(100), weights), xent)
    #weighted_xent = xent

    total_loss = - tf.div(tf.reduce_sum(weighted_xent), num_labels, name='value')

    tf.add_to_collection('_losses', total_loss)
    return total_loss
