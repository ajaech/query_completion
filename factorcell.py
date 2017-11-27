import tensorflow as tf


class FactorCell(tf.nn.rnn_cell.RNNCell):
  """LSTM cell with coupled input and forget gates."""
  
  def Lock(self):
    """Used to lock in the personalization."""
    lock_ops = [tf.no_op()]

    if self.lowrank_adaptation:
      # compute the new W
      left_adapt = tf.squeeze(self.left_adapt)
      right_adapt = tf.squeeze(self.right_adapt)
      final_w = self.W + tf.matmul(left_adapt, right_adapt)
      lock_ops.append(self.lockedW.assign(final_w))
    else:
      lock_ops.append(self.lockedW.assign(self.W))

    if self.mikolov_adapt:
      final_bias = tf.squeeze(self.bias + self.delta)
      lock_ops.append(self.lockedBias.assign(final_bias))
    else:
      lock_ops.append(self.lockedBias.assign(self.bias))

    self.lock_op = tf.group(*lock_ops)


  def __init__(self, num_units, embedding_size, context_embed, 
               bias_adaptation=False, lowrank_adaptation=False,
               rank=10, layer_norm=False, dropout_keep_prob=None):
    self._num_units = num_units
    self._forget_bias = 1.0
    self._activation = tf.tanh
    self.mikolov_adapt = bias_adaptation
    self.lowrank_adaptation = lowrank_adaptation
    self.layer_norm = layer_norm
    self._keep_prob = dropout_keep_prob

    input_size = num_units + embedding_size

    with tf.variable_scope('factor_cell'):
      self.W = tf.get_variable('W', [input_size, 3 * self._num_units])
      self.lockedW = tf.Variable(tf.zeros_like(self.W), name='lockedW',
                                 collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                 trainable=False)

      self.bias = tf.get_variable('bias', [3 * self._num_units],
                                  initializer=tf.constant_initializer(0.0, tf.float32))
      self.lockedBias = tf.Variable(tf.zeros_like(self.bias), name='lockedBias',
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                    trainable=False)

      if self.layer_norm:
        self.gammas = []
        self.betas = []
        for gate in ['j', 'f', 'o']:
          self.gammas.append(
            tf.get_variable('gamma_' + gate, shape=[num_units],
                            initializer=tf.constant_initializer(1.0)))
          self.betas.append(
            tf.get_variable('beta_' + gate, shape=[num_units],
                            initializer=tf.constant_initializer(0.0)))

      if self.lowrank_adaptation:
        context_embed_size = context_embed.get_shape()[1].value
        left_adapt_generator = tf.get_variable(
          'left_generator', [context_embed_size, input_size * rank])
        left_adapt_unshaped = tf.matmul(context_embed, left_adapt_generator, 
                                        name='left_matmul')
        self.left_adapt = tf.reshape(
          left_adapt_unshaped, [-1, input_size, rank])

        right_adapt_generator = tf.get_variable(
          'right_generator', [context_embed_size, 3 * num_units * rank])
        right_adapt_unshaped = tf.matmul(context_embed, right_adapt_generator, 
                                         name='right_matmul')
        self.right_adapt = tf.reshape(
          right_adapt_unshaped, [-1, rank, 3 * num_units])

      if self.mikolov_adapt:
        context_embed_size = context_embed.get_shape()[1].value
        self.biases = tf.get_variable(
          'mikolov_biases', [context_embed_size, 3 * self._num_units])
        self.delta = tf.matmul(context_embed, self.biases)
      self.Lock()

  def __str__(self):
    return 'factor cell of size {0}'.format(self._num_units)

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)
    
  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None, reuse=None, use_locked=False):
    with tf.variable_scope("hyper_lstm_cell", reuse=reuse):
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = state
      the_input = tf.concat(axis=1, values=[inputs, h])
      
      if use_locked:
        result = tf.matmul(the_input, self.lockedW)
      else:
        result = tf.matmul(the_input, self.W)

      if self.lowrank_adaptation and not use_locked:
        input_expanded = tf.expand_dims(the_input, 1)
        intermediate = tf.matmul(input_expanded, self.left_adapt)
        final = tf.matmul(intermediate, self.right_adapt)
        result += tf.squeeze(final)
      if self.mikolov_adapt and not use_locked:
        result += self.delta
      
      if use_locked:
        result += self.lockedBias
      else:
        result += self.bias

      # j = new_input, f = forget_gate, o = output_gate
      j, f, o = tf.split(axis=1, num_or_size_splits=3, value=result)

      def Norm(inputs, gamma, beta):
        m, v = tf.nn.moments(inputs, [1], keep_dims=True)
        normalized_input = (inputs - m) / tf.sqrt(v + 1e-5)
        return normalized_input * gamma + beta

      if self.layer_norm:     
        j = Norm(j, self.gammas[0], self.betas[0])
        f = Norm(f, self.gammas[1], self.betas[1])
        o = Norm(o, self.gammas[2], self.betas[2])

      g = self._activation(j)

      # recurrent dropout without memory loss
      if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
        g = tf.nn.dropout(g, self._keep_prob)

      forget_gate = tf.sigmoid(f + self._forget_bias)
      input_gate = 1.0 - forget_gate  # input and forget gates are coupled

      new_c = (c * forget_gate + input_gate * g)
      new_h = self._activation(new_c) * tf.sigmoid(o)

      new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
      return new_h, new_state
