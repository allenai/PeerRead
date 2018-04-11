"""
 three prediction models: RNN, DAN, CNN
"""

import tensorflow as tf


class RNN(object):
  def __init__(self, config, binarize=False, embedding = None):
    self.config = config
    self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
    self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    self.binarize = binarize
    self.out_layers = 1

    if embedding is not None:
      self.embedding = tf.get_variable(
          'embedding', [self.config.vocab_size, self.config.embedding_dim],
          dtype=tf.float32, trainable=False)
      self.embedding_placeholder = tf.placeholder(tf.float32,
          [self.config.vocab_size, self.config.embedding_dim])
      self.embedding_init = self.embedding.assign(self.embedding_placeholder)
    else:
      self.embedding = tf.get_variable('embedding',
          [self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32)

    self.rnn()


  def rnn(self):
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)
    def gru_cell():
      return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

    def dropout():
      if (self.config.rnn == 'lstm'):
        cell = lstm_cell()
      else:
        cell = gru_cell()
      return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    with tf.device('/cpu:0'):
      embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

    with tf.name_scope("rnn"):
      cells = [dropout() for _ in range(self.config.num_layers)]
      rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
      _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
      last = _outputs[:, -1, :]

    with tf.name_scope("score"):
      fc = last
      for ol in range(self.out_layers - 1):
        fc = tf.layers.dense(fc, self.config.hidden_dim, name='fc%d'%(ol))
        fc = tf.contrib.layers.dropout(fc, self.keep_prob)
        fc = tf.nn.relu(fc)
      self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc%d'%(self.out_layers-1))

    with tf.name_scope("optimize"):
      self.loss = tf.reduce_mean(tf.square(self.logits - self.input_y))
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5.0)
      optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate, decay=0.9)
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


class CNN(object):
  def __init__(self, config, binarize=False, embedding = None):
    self.config = config
    self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
    self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


    self.binarize = binarize
    self.out_layers = 1

    if embedding is not None:
      self.embedding = tf.get_variable(
          'embedding', [self.config.vocab_size, self.config.embedding_dim],
          dtype=tf.float32, trainable=False)
      self.embedding_placeholder = tf.placeholder(tf.float32,
          [self.config.vocab_size, self.config.embedding_dim])
      self.embedding_init = self.embedding.assign(self.embedding_placeholder)
    else:
      self.embedding = tf.get_variable('embedding',
          [self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32)

    self.cnn()

  def cnn(self):
    with tf.device('/cpu:0'):
      embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

    with tf.name_scope("cnn"):
      conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
      gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

    with tf.name_scope("score"):

      fc = gmp
      for ol in range(self.out_layers - 1):
        fc = tf.layers.dense(fc, self.config.hidden_dim, name='fc%d'%(ol))
        fc = tf.contrib.layers.dropout(fc, self.keep_prob)
        fc = tf.nn.relu(fc)
      self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc%d'%(self.out_layers-1))

    with tf.name_scope("optimize"):
      self.loss = tf.reduce_mean(tf.square(self.logits - self.input_y))
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5.0)
      optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate, decay=0.9)
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)




class DAN(object):
  def __init__(self, config, binarize=False, embedding = None):
    self.config = config
    self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
    self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


    self.binarize = binarize
    self.out_layers = 1


    if embedding is not None:
      self.embedding = tf.get_variable(
          'embedding', [self.config.vocab_size, self.config.embedding_dim],
          dtype=tf.float32, trainable=True)
      self.embedding_placeholder = tf.placeholder(tf.float32,
          [self.config.vocab_size, self.config.embedding_dim])
      self.embedding_init = self.embedding.assign(self.embedding_placeholder)
    else:
      self.embedding = tf.get_variable('embedding',
          [self.config.vocab_size, self.config.embedding_dim], dtype=tf.float32)


    self.dan()

  def dan(self):
    with tf.device('/cpu:0'):
      embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

    with tf.name_scope("dan"):
      last = tf.reduce_mean(embedding_inputs, 1, name='last_dan')

    with tf.name_scope("score"):
      fc = last
      for ol in range(self.out_layers - 1):
        fc = tf.layers.dense(fc, self.config.hidden_dim, name='fc%d'%(ol))
        fc = tf.contrib.layers.dropout(fc, self.keep_prob)
        fc = tf.nn.relu(fc)
      self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc%d'%(self.out_layers-1))

    with tf.name_scope("optimize"):

      self.loss = tf.reduce_mean(tf.square(self.logits - self.input_y))
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5.0)
      optimizer = tf.train.RMSPropOptimizer(self.config.learning_rate, decay=0.9)
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)



