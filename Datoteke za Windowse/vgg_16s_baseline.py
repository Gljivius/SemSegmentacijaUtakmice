import tensorflow as tf
import slim
from slim import ops
from slim import scopes
import numpy as np
import losses

FLAGS = tf.app.flags.FLAGS

def conv_layer(bottom, name, dillate):
  filt = get_conv_filter(name)
  if (dillate == False):
    conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
  else:
    conv = tf.nn.atrous_conv2d(bottom, filt, rate = 2, padding='SAME')
  conv_biases = np.load('vgg16/' + name + '_b.npy')
  bias = tf.nn.bias_add(conv, conv_biases)
  relu = tf.nn.relu(bias)
    
  return relu

def get_conv_filter(name):
  tezine = np.load('vgg16/' + name + '_W.npy')
  init = tf.constant_initializer(tezine, dtype=tf.float32)
  shape = tezine.shape
  var = tf.get_variable(name="filter-" + name, initializer=init, shape=shape)
  
  return var

def inference(inputs, is_training=True):
  conv1_sz = 64
  conv2_sz = 128
  conv3_sz = 256
  conv4_sz = 512
  conv5_sz = 512
  k = 3

  with scopes.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.0005):
  #with scopes.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01, weight_decay=0.005):
  #with scopes.arg_scope([slim.ops.conv2d, slim.ops.fc], stddev=0.01):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.dropout],
                          is_training=is_training):
      #net = slim.ops.repeat_op(1, inputs, slim.ops.conv2d, 64, [3, 3], scope='conv1')
      net = conv_layer(inputs, 'conv1_1', dillate = False)
      net = conv_layer(net, 'conv1_2', dillate = False)
      net = ops.max_pool(net, [2, 2], scope='pool1')
      net = conv_layer(net, 'conv2_1', dillate = False)
      net = conv_layer(net, 'conv2_2', dillate = False)
      net = ops.max_pool(net, [2, 2], scope='pool2')
      net = conv_layer(net, 'conv3_1', dillate = False)
      net = conv_layer(net, 'conv3_2', dillate = False)
      net = conv_layer(net, 'conv3_3', dillate = False)
      net = ops.max_pool(net, [2, 2], scope='pool3')
      net = conv_layer(net, 'conv4_1', dillate = False)
      net = conv_layer(net, 'conv4_2', dillate = False)
      net = conv_layer(net, 'conv4_3', dillate = False)
      net = ops.max_pool(net, [2, 2], scope='pool4')
      #net = Convolve(net, conv5_sz, k, 'conv5_1', vgg_layers)
      #net = Convolve(net, conv5_sz, k, 'conv5_2', vgg_layers)
      #conv5_3 = Convolve(net, conv5_sz, k, 'conv5_3', vgg_layers)
      #pool5 = ops.max_pool(conv5_3, [2, 2], scope='pool5')

      #conv3_shape = conv3_3.get_shape()
      #resize_shape = [conv3_shape[1].value, conv3_shape[2].value]
      #up_conv5_3 = tf.image.resize_nearest_neighbor(conv5_3, resize_shape)
      ##up_conv5_3 = tf.image.resize_nearest_neighbor(conv5_3, [108, 256])
      #concat = tf.concat(3, [conv3_3, up_conv5_3])
      #net = slim.ops.max_pool(net, [2, 2], scope='pool5')
      with scopes.arg_scope([ops.conv2d, ops.fc]):
        net = ops.conv2d(net, 1024, [1, 1], scope='fc6')
        #net = ops.conv2d(net, 1024, [1, 1], scope='fc7')
      #net = ops.conv2d(net, 1024, [1, 1], scope='fc6')
      #net = ops.conv2d(net, 1024, [1, 1], scope='fc7')
      net = ops.conv2d(net, FLAGS.num_classes, [1, 1], activation=None, scope='score')
      #net = slim.ops.flatten(net, scope='flatten5')
      #net = slim.ops.fc(net, 4096, scope='fc6')
      #net = slim.ops.dropout(net, 0.5, scope='dropout6')
      #net = slim.ops.fc(net, 4096, scope='fc7')
      #net = slim.ops.dropout(net, 0.5, scope='dropout7')
      #net = slim.ops.fc(net, 1000, activation=None, scope='fc8')
      logits_up = tf.image.resize_bilinear(net, [FLAGS.img_height, FLAGS.img_width],
                                           name='resize_scores')

  #var_list = tf.all_variables()
  #var_list = slim.variables.get_variables()
  #var_map = {}
  #for v in var_list:
  #  print(v.name)
  #  var_map[v.name] = v
  #var_map['conv1_1/weights:0'].assign(vgg_layers['conv1_1'][0])
  #init_vgg(vgg_layers, vgg_layer_names, var_map)
  #init_op = tf.initialize_variables([var_map['fc6/weights:0'], var_map['fc6/biases:0'],
  #  var_map['fc7/weights:0'], var_map['fc7/biases:0'], var_map['score/weights:0'],
  #  var_map['score/biases:0'], var_map['global_step:0']])

  #print(var_list[1].name)
  #var_list[1].assign(vgg_layers['conv1_1'])
  return logits_up


def loss(logits, labels, num_labels, apriori, bayes = False, is_training=True):
  if (bayes is True):
    loss_val = losses.bayesian_loss(logits, labels, num_labels, apriori)
  else:
    loss_val = losses.cross_entropy_loss(logits, labels, num_labels)
  all_losses = [loss_val]
  #all_losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)

  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)

  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)

  return total_loss
  
  #return losses.weighted_hinge_loss(logits, labels, weights, num_labels)
  #return losses.cross_entropy_loss(logits, labels, num_labels)

