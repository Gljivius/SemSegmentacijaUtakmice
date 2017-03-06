import os
import time
import tensorflow as tf

import numpy as np
import skimage as ski
import skimage.data, skimage.transform

import train_helper
import eval_helper

import vgg_16s_baseline as model

SAVE_DIR = os.path.join('Log/', train_helper.get_time_string())
RGB_DIR = '/home/ivan/SemSegmentacija/Slike/SlikeZaUcenje/'      #unesite ovdje svoj put do datoteka
GT_DIR = '/home/ivan/SemSegmentacija/Slike/SlikeOznaka/'         #unesite ovdje svoj put do datoteka
OUTPUT_DIR = '/home/ivan/SemSegmentacija/Slike/IzlazneSlike/'    #unesite ovdje svoj put do datoteka
IMG_WIDTH = 1280
IMG_HEIGHT = 738
VGG_MEAN = [123.68, 116.779, 103.939]

CLASS_INFO = [[128,64,128,   'TeamA'],
              [244,35,232,   'TeamB'],
              [70,70,70,     'Terrain'],
              [0,0,255,      'Crowd'],
              [255,0,0,      'Other personel']]

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', SAVE_DIR, \
    """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('output_dir', OUTPUT_DIR, '')
tf.app.flags.DEFINE_integer('img_width', IMG_WIDTH, '')
tf.app.flags.DEFINE_integer('img_height', IMG_HEIGHT, '')
tf.app.flags.DEFINE_integer('img_depth', 3, '')

tf.app.flags.DEFINE_string('data_rgb_dir', RGB_DIR, '')
tf.app.flags.DEFINE_string('data_gt_dir', GT_DIR, '')
tf.app.flags.DEFINE_string('debug_dir', os.path.join(FLAGS.train_dir, 'debug'), '')
tf.app.flags.DEFINE_string('vgg_init_dir', '/home/kivan/datasets/pretrained/vgg16/', '')
#tf.app.flags.DEFINE_integer('max_steps', 100000,
#                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_epochs', 20, 'Number of epochs to run.')
tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_integer('num_classes',5, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')

# 1e-4 best, 1e-3 is too big
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 3.0,
#tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
#tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.2,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, '')


class Trainer(object):
  def __init__(self):
    self.rgb_data = []
    self.label_data = []
    self.filenames = []
    self.data_num_labels = []
    self.CLASS_APRIORI = np.array([0, 0, 0, 0, 0, 0])
    
    #with tf.Graph().as_default(), tf.device('/gpu:0'):
    with tf.Graph().as_default():
      #sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
      config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
      #config.gpu_options.per_process_gpu_memory_fraction = 0.5 # don't hog all vRAM
      #config.operation_timeout_in_ms=5000   # terminate on long hangs
      sess = tf.Session(config=config)
      # Create a variable to count the number of train() calls. This equals the
      # number of batches processed * FLAGS.num_gpus.
      global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                    trainable=False)

      # Calculate the learning rate schedule.
      #decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
      decay_steps = 1e10

      # Decay the learning rate exponentially based on the number of steps.
      lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True)

      data_shape = (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth)
      labels_shape = (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width)
      self.image = tf.placeholder(tf.float32, shape=data_shape)
      self.labels = tf.placeholder(tf.int32, shape=labels_shape)
      self.num_labels = tf.placeholder(tf.float32, shape=())
      self.apriori = tf.placeholder(tf.float32, shape=(6,))
      # Build a Graph that computes the logits predictions from the inference model.
      # Calculate loss.
      with tf.variable_scope("model"):
        self.logits = model.inference(self.image)
        self.loss = model.loss(self.logits, self.labels, self.num_labels, self.apriori, bayes = True)
      with tf.variable_scope("model", reuse=True):
        self.logits_valid = model.inference(self.image)
        self.loss_valid = model.loss(self.logits, self.labels, self.num_labels, self.apriori, bayes = True, is_training=False)

      # Add a summary to track the learning rate.
      tf.summary.scalar('learning_rate', lr)
      #tf.summary.scalar('learning_rate', tf.mul(lr, tf.constant(1 / FLAGS.initial_learning_rate)))

      #with tf.control_dependencies([loss_averages_op]):
      #opt = tf.train.GradientDescentOptimizer(lr)
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(self.loss)

      # Apply gradients.
      apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

      # Add histograms for trainable variables.
      for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
      # Add histograms for gradients.
      for grad, var in grads:
        if grad is not None:
          tf.summary.histogram(var.op.name + '/gradients', grad)
      #grad = grads[-2][0]
      #print(grad)

      # Track the moving averages of all trainable variables.
      variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
      variables_averages_op = variable_averages.apply(tf.trainable_variables())

      # with slim's BN
      #batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
      #batchnorm_updates_op = tf.group(*batchnorm_updates)
      #train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)
      with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        self.train_op = tf.no_op(name='train')

      # Create a saver.
      self.saver = tf.train.Saver()
      #saver = tf.train.Saver(tf.all_variables())
      resume_path = None
      if resume_path is None:
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        sess.run(init)
      else:
        assert tf.gfile.Exists(resume_path)
        self.saver.restore(sess, resume_path)
        #variables_to_restore = tf.get_collection(
        #    slim.variables.VARIABLES_TO_RESTORE)
        #restorer = tf.train.Saver(variables_to_restore)
        #restorer.restore(sess, resume_path)

      # Build the summary operation based on the TF collection of Summaries.
      #summary_op = tf.merge_all_summaries()
      #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)
    self.global_step = global_step
    self.data_shape = data_shape
    self.labels_shape = labels_shape
    self.sess = sess
    self.step = 1
    self._load_existing_data()
    self.CLASS_APRIORI = 1 / (self.CLASS_APRIORI / np.sum(self.CLASS_APRIORI))
    np.clip(self.CLASS_APRIORI, 0, 50, out = self.CLASS_APRIORI)

  def _load_existing_data(self):
    files = next(os.walk(FLAGS.data_rgb_dir))[2]
    for f in files:
      self.add_image(f)
  
  def loadWeights(self):
    self.saver.restore(self.sess, '/home/ivan/Desktop/interactive/Spremljeno/model.ckpt')
    print("Model restored.")
  
  def saveWeights(self):
    self.saver.save(self.sess, '/home/ivan/Desktop/interactive/Spremljeno/model.ckpt')
    print("Model saved.")

  def _train_iter(self, img_num):
    #conf_mat = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64)
    #conf_mat = np.ascontiguousarray(conf_mat)
    start_time = time.time()
    run_ops = [self.train_op, self.loss, self.logits, self.global_step]
    #if step % 100 == 0:
    #  run_ops += [self.summary_op]
    #  ret_val = sess.run(run_ops)
    #  (_, loss_val, scores, yt, global_step_val, summary_str) = ret_val
    #  self.summary_writer.add_summary(summary_str, global_step_val)
    #else:
    img = self.rgb_data[img_num]
    labels = self.label_data[img_num]
    num_labels = self.data_num_labels[img_num]
    ret_val = self.sess.run(run_ops, feed_dict={self.image : img, self.labels : labels,
                                                self.num_labels : num_labels, self.apriori : self.CLASS_APRIORI})
    (_, loss_val, scores, global_step_val) = ret_val

    duration = time.time() - start_time
    num_examples_per_step = FLAGS.batch_size
    examples_per_sec = num_examples_per_step / duration
    sec_per_batch = float(duration)

    format_str = 'step %d / %d, loss = %.6f (%.1f examples/sec; %.3f sec/batch)'
    #print('lr = ', clr)
    print(format_str % (self.step, len(self.filenames), loss_val,
                        examples_per_sec, sec_per_batch))
    self.step += 1


  def train(self, num_epochs):
    for num in range(num_epochs):
      for i, f in enumerate(self.filenames):
        self._train_iter(i)


  def infer(self, img_num):
    for img_num, f in enumerate(self.filenames):
      start_time = time.time()
      run_ops = [self.loss_valid, self.logits_valid]
      img = self.rgb_data[img_num]
      labels = self.label_data[img_num]
      num_labels = self.data_num_labels[img_num]
      ret_val = self.sess.run(run_ops, feed_dict={self.image : img, self.labels : labels,
                                                  self.num_labels : num_labels, self.apriori : self.CLASS_APRIORI})
      (loss_val, out_logits) = ret_val

      duration = time.time() - start_time
      sec_per_batch = float(duration)
      print('time = ', sec_per_batch)

      print('loss = ', loss_val)
      y = out_logits[0].argmax(2).astype(np.int32)
      eval_helper.draw_output(y, CLASS_INFO, os.path.join(FLAGS.output_dir, self.filenames[img_num]))



  def add_image(self, filename):
    print(filename)
    rgb_path = os.path.join(FLAGS.data_rgb_dir, filename)
    label_path = os.path.join(FLAGS.data_gt_dir, filename)
    width = FLAGS.img_width
    height = FLAGS.img_height

    rgb_img = ski.data.load(rgb_path)
    rgb_img = ski.transform.resize(rgb_img, (height, width), preserve_range=True, order=3)
    rgb_img = rgb_img.astype(np.float32)
    for c in range(3):
      rgb_img[:,:,c] -= VGG_MEAN[c]
    rgb_img = rgb_img.reshape(1, height, width, 3)
    labels = ski.data.load(label_path)
    labels = ski.transform.resize(labels, (height, width), preserve_range=True, order=0)
    labels = labels.astype(np.int32)
    labels = labels.reshape(1, height, width)
    num_labels = (labels < 255).sum()
    
    self.CLASS_APRIORI[0] += (labels == 0).sum()
    self.CLASS_APRIORI[1] += (labels == 30).sum()
    self.CLASS_APRIORI[2] += (labels == 60).sum()
    self.CLASS_APRIORI[3] += (labels == 90).sum()
    self.CLASS_APRIORI[4] += (labels == 120).sum()
    self.CLASS_APRIORI[5] += (labels == 150).sum()
    
    self.rgb_data += [rgb_img]
    self.label_data += [labels]
    self.filenames += [filename]
    self.data_num_labels += [num_labels]


  def num_examples(self):
    return len(self.filenames)


  def get_filenames(self):
    return self.filenames
