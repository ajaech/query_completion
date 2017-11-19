import tensorflow as tf
from factorcell import FactorCell


class Model(object):
    
    def __init__(self, params, training_mode=True):
        self.params = params
        self.BuildGraph(params, training_mode=training_mode)
        if not training_mode:
          self.BuildDecoderGraph()
        
    def BuildGraph(self, params, training_mode=True):
        self.queries = tf.placeholder(tf.int32, [None, params.max_len])
        self.query_lengths = tf.placeholder(tf.int32, [None])
        self.user_ids = tf.placeholder(tf.int32, [None])

        x = self.queries[:, :-1]
        y = self.queries[:, 1:]

        self.char_embeddings = tf.get_variable(
            'char_embeddings', [params.vocab_size, params.char_embed_size])
        self.char_bias = tf.get_variable('char_bias', [params.vocab_size])
        self.user_embed_mat = tf.get_variable(
            'user_embed_mat', [params.user_vocab_size, params.user_embed_size])

        inputs = tf.nn.embedding_lookup(self.char_embeddings, x)
        
        indicator = tf.sequence_mask(tf.to_int32(self.query_lengths - 1), 
                                     params.max_len - 1)
        _mask = tf.where(indicator, tf.ones_like(x, dtype=tf.float32),
                         tf.zeros_like(x, dtype=tf.float32))
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, (), name='keep_prob')

        self.user_embeddings = tf.nn.embedding_lookup(self.user_embed_mat, self.user_ids)

        with tf.variable_scope('rnn'):
            self.decoder_cell = FactorCell(params.num_units, params.char_embed_size,
                                           self.user_embeddings,
                                           mikilovian_adaptation=params.use_mikolov_adaptation,
                                           lowrank_adaptation=params.use_lowrank_adaptation,
                                           rank=params.rank,
                                           layer_norm=params.use_layer_norm,
                                           dropout_keep_prob=self.dropout_keep_prob)
        
            outputs, _ = tf.nn.dynamic_rnn(self.decoder_cell, inputs,
                                           sequence_length=self.query_lengths,
                                           dtype=tf.float32)
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

        self.per_word_loss = tf.reshape(masked_loss, tf.shape(x))
        self.per_sentence_loss = tf.div(tf.reduce_sum(self.per_word_loss, 1),
                                        tf.reduce_sum(_mask, 1))
        self.per_sentence_loss = tf.reduce_sum(self.per_word_loss, 1)

        total_loss = tf.reduce_sum(masked_loss)
        self.words_in_batch = tf.to_float(tf.reduce_sum(self.query_lengths - 1))
        self.avg_loss = total_loss / self.words_in_batch
        
        if training_mode:
          optimizer = tf.train.AdamOptimizer(0.001)
          self.train_op = optimizer.minimize(self.avg_loss)

    def BuildDecoderGraph(self):
        self.prev_word = tf.placeholder(tf.int32, [None], name='prev_word')
        self.prev_c = tf.placeholder(tf.float32, [None, self.params.num_units], 
                                     name='prev_c')
        self.prev_h = tf.placeholder(tf.float32, [None, self.params.num_units], 
                                     name='prev_h')
        self.temperature = tf.placeholder_with_default([1.0], [1])
        
        prev_embed = tf.nn.embedding_lookup(self.char_embeddings, self.prev_word)
        
        state = tf.nn.rnn_cell.LSTMStateTuple(self.prev_c, self.prev_h)
        result, (self.next_c, self.next_h) = self.decoder_cell(prev_embed, state,
                                                               use_locked=True)
            
        with tf.variable_scope('rnn', reuse=True):
            proj_result = tf.layers.dense(result, self.params.char_embed_size,
                                          reuse=True, name='proj')
        logits = tf.matmul(proj_result, self.char_embeddings, 
                           transpose_b=True) + self.char_bias    
        self.beam_size = tf.placeholder_with_default(1, (), name='beam_size')
        self.next_prob = tf.nn.softmax(logits / self.temperature)
        self.selected_p, self.selected = tf.nn.top_k(self.next_prob, self.beam_size)
