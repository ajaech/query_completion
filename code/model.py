import os
import tensorflow as tf
from factorcell import FactorCell
from vocab import Vocab
import helper


class MetaModel(object):
  """Helper class for loading models."""

  def __init__(self, expdir):
    self.expdir = expdir
    self.params = helper.GetParams(os.path.join(expdir, 'char_vocab.pickle'), 'eval', 
                                   expdir)
    # mapping of characters to indices
    self.char_vocab = Vocab.Load(os.path.join(expdir, 'char_vocab.pickle'))
    # mapping of user ids to indices
    self.user_vocab = Vocab.Load(os.path.join(expdir, 'user_vocab.pickle'))
    self.params.vocab_size = len(self.char_vocab)
    self.params.user_vocab_size = len(self.user_vocab)

    # construct the tensorflow graph
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.model = Model(self.params, training_mode=False)
      self.char_tensor = tf.constant(self.char_vocab.GetWords(), name='char_tensor')
      self.beam_chars = tf.nn.embedding_lookup(self.char_tensor, self.model.selected)

  def Lock(self, user_id=0):
    """Locking precomputes the adaptation for a given user."""
    self.session.run(self.model.decoder_cell.lock_op,
                     {self.model.user_ids: [user_id]})

  def MakeSession(self, threads=8):
    """Create the session with the given number of threads."""
    config = tf.ConfigProto(inter_op_parallelism_threads=threads,
                            intra_op_parallelism_threads=threads)
    with self.graph.as_default():
      self.session = tf.Session(config=config)

  def Restore(self):
    """Initialize all variables and restore model from disk."""
    with self.graph.as_default():
      saver = tf.train.Saver(tf.trainable_variables())
      self.session.run(tf.global_variables_initializer())
      saver.restore(self.session, os.path.join(self.expdir, 'model.bin'))

  def MakeSessionAndRestore(self, threads=8):
    self.MakeSession(threads)
    self.Restore()


class Model(object):
  """Defines the Tensorflow graph for training and testing a model."""

  def __init__(self, params, training_mode=True, optimizer=tf.train.AdamOptimizer,
               learning_rate=0.001):
    self.params = params
    opt = optimizer(learning_rate)
    self.BuildGraph(params, training_mode=training_mode, optimizer=opt)
    if not training_mode:
      self.BuildDecoderGraph()

  def BuildGraph(self, params, training_mode=True, optimizer=None):
    self.queries = tf.placeholder(tf.int32, [None, params.max_len], name='queries')
    self.query_lengths = tf.placeholder(tf.int32, [None], name='query_lengths')
    self.user_ids = tf.placeholder(tf.int32, [None], name='user_ids')

    x = self.queries[:, :-1]  # strip off the end of query token
    y = self.queries[:, 1:]   # need to predict y from x

    self.char_embeddings = tf.get_variable(
        'char_embeddings', [params.vocab_size, params.char_embed_size])
    self.char_bias = tf.get_variable('char_bias', [params.vocab_size])
    self.user_embed_mat = tf.get_variable(  # this defines the user embeddings
        'user_embed_mat', [params.user_vocab_size, params.user_embed_size])

    inputs = tf.nn.embedding_lookup(self.char_embeddings, x)

    # create a mask to zero out the loss for the padding tokens
    indicator = tf.sequence_mask(self.query_lengths - 1, params.max_len - 1)
    _mask = tf.where(indicator, tf.ones_like(x, dtype=tf.float32),
                     tf.zeros_like(x, dtype=tf.float32))
    self.dropout_keep_prob = tf.placeholder_with_default(1.0, (), name='keep_prob')

    user_embeddings = tf.nn.embedding_lookup(self.user_embed_mat, self.user_ids)

    self.use_time_features = False
    if hasattr(params, 'use_time_features') and params.use_time_features:
      self.use_time_features = True
      self.dayofweek = tf.placeholder(tf.int32, [None], name='dayofweek')
      self.hourofday = tf.placeholder(tf.int32, [None], name='hourofday')
      self.day_embed_mat = tf.get_variable('day_embed_mat', [7, 2])
      self.hour_embed_mat = tf.get_variable('hour_embed_mat', [24, 3])

      hour_embeds = tf.nn.embedding_lookup(self.hour_embed_mat, self.hourofday)
      day_embeds = tf.nn.embedding_lookup(self.day_embed_mat, self.dayofweek)
      
      user_embeddings = tf.concat(axis=1, values=[user_embeddings, hour_embeds, day_embeds])

    with tf.variable_scope('rnn'):
      self.decoder_cell = FactorCell(params.num_units, params.char_embed_size,
                                     user_embeddings,
                                     bias_adaptation=params.use_mikolov_adaptation,
                                     lowrank_adaptation=params.use_lowrank_adaptation,
                                     rank=params.rank,
                                     layer_norm=params.use_layer_norm,
                                     dropout_keep_prob=self.dropout_keep_prob)

      outputs, _ = tf.nn.dynamic_rnn(self.decoder_cell, inputs,
                                     sequence_length=self.query_lengths,
                                     dtype=tf.float32)
      # reshape outputs to 2d before passing to the output layer
      reshaped_outputs = tf.reshape(outputs, [-1, params.num_units])
      projected_outputs = tf.layers.dense(reshaped_outputs, params.char_embed_size,
                                          name='proj')
      reshaped_logits = tf.matmul(projected_outputs, self.char_embeddings, 
                                  transpose_b=True) + self.char_bias

    reshaped_labels = tf.reshape(y, [-1])
    reshaped_mask = tf.reshape(_mask, [-1])

    reshaped_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=reshaped_logits, labels=reshaped_labels)
    masked_loss = tf.multiply(reshaped_loss, reshaped_mask)

    # reshape the loss back to the input size in order to compute
    # the per sentence loss
    self.per_word_loss = tf.reshape(masked_loss, tf.shape(x))
    self.per_sentence_loss = tf.div(tf.reduce_sum(self.per_word_loss, 1),
                                    tf.reduce_sum(_mask, 1))
    self.per_sentence_loss = tf.reduce_sum(self.per_word_loss, 1)

    total_loss = tf.reduce_sum(masked_loss)
    self.words_in_batch = tf.to_float(tf.reduce_sum(self.query_lengths - 1))
    self.avg_loss = total_loss / self.words_in_batch

    if training_mode:
      self.train_op = optimizer.minimize(self.avg_loss)

  def BuildDecoderGraph(self):
    """This part of the graph is only used for evaluation.
    
    It computes just a single step of the LSTM.
    """
    self.prev_word = tf.placeholder(tf.int32, [None], name='prev_word')
    self.prev_hidden_state = tf.placeholder(
      tf.float32, [None, 2 * self.params.num_units], name='prev_hidden_state')
    prev_c = self.prev_hidden_state[:, :self.params.num_units]
    prev_h = self.prev_hidden_state[:, self.params.num_units:]

    # temperature can be used to tune diversity of the decoding
    self.temperature = tf.placeholder_with_default([1.0], [1])

    prev_embed = tf.nn.embedding_lookup(self.char_embeddings, self.prev_word)

    state = tf.nn.rnn_cell.LSTMStateTuple(prev_c, prev_h)
    result, (next_c, next_h) = self.decoder_cell(
      prev_embed, state, use_locked=True)
    self.next_hidden_state = tf.concat([next_c, next_h], 1)

    with tf.variable_scope('rnn', reuse=True):
      proj_result = tf.layers.dense(
        result, self.params.char_embed_size, reuse=True, name='proj')
    logits = tf.matmul(proj_result, self.char_embeddings, 
                       transpose_b=True) + self.char_bias
    prevent_unk = tf.one_hot([0], self.params.vocab_size, -30.0)
    self.next_prob = tf.nn.softmax(prevent_unk + logits / self.temperature)
    self.next_log_prob = tf.nn.log_softmax(logits / self.temperature)
    
    # return the top `beam_size` number of characters for use in decoding
    self.beam_size = tf.placeholder_with_default(1, (), name='beam_size')
    log_probs, self.selected = tf.nn.top_k(self.next_log_prob, self.beam_size)
    self.selected_p = -log_probs  # cost is the negative log likelihood
