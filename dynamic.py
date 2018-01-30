import argparse
import os
import numpy as np
import tensorflow as tf
import time
import sys
from beam import GetCompletions
from dataset import LoadData, Dataset
from helper import GetPrefixLen
from metrics import GetRankInList, MovingAvg
from model import MetaModel


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('expdir', help='experiment directory')
  parser.add_argument('--data', type=str, action='append', dest='data',
                      help='where to load the data')
  parser.add_argument('--optimizer', default='sgd', 
                       choices=['sgd', 'adam', 'ada'],
                       help='which optimizer to use to learn user embeddings')
  parser.add_argument('--learning_rate', type=float, default=None)
  parser.add_argument('--threads', type=int, default=12,
                      help='how many threads to use in tensorflow')
  parser.add_argument('--tuning', action='store_true', dest='tuning',
                      help='when tuning don\'t do beam search decoding',
                      default=False)
  parser.add_argument('--limit', type=int, default=385540, 
                       help='how many queries to evaluate')
  args = parser.parse_args()


class DynamicModel(MetaModel):
    
  def __init__(self, expdir, learning_rate=None, threads=8,
               optimizer=tf.train.GradientDescentOptimizer):
    super(DynamicModel, self).__init__(expdir)

    if learning_rate is None:
      if self.params.use_lowrank_adaptation:
        learning_rate = 0.22
      else:
        learning_rate = 1.0

    self.MakeSession(threads)
    self.Restore()
    with self.graph.as_default():
      unk_embed = self.model.user_embed_mat.eval(
        session=self.session)[self.user_vocab['<UNK>']]
      self.reset_user_embed = tf.scatter_update(
          self.model.user_embed_mat, [0], np.expand_dims(unk_embed, 0))
      self.session.run(self.reset_user_embed)

      with tf.variable_scope('optimizer'):
        self.train_op = tf.no_op()            
        if (self.params.use_lowrank_adaptation or 
            self.params.use_mikolov_adaptation):
          self.train_op = optimizer(learning_rate).minimize(
            self.model.avg_loss, var_list=[self.model.user_embed_mat])
      opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "optimizer")
      if len(opt_vars):
        self.reset_user_embed = tf.group(self.reset_user_embed,
                                         tf.variables_initializer(opt_vars))

  def Train(self, query):
    qIds = np.zeros((1, self.params.max_len))
    for i in range(min(self.params.max_len, len(query))):
      qIds[0, i] = self.char_vocab[query[i]]
        
    feed_dict = {
      self.model.user_ids: np.array([0]),
      self.model.query_lengths: np.array([len(query)]),
      self.model.queries: qIds
    }
        
    c, words_in_batch, _ = self.session.run(
        [self.model.avg_loss, self.model.words_in_batch, self.train_op], 
        feed_dict)
    return c, words_in_batch

if __name__ == '__main__':
  optimizer = {'sgd': tf.train.GradientDescentOptimizer,
               'adam': tf.train.AdamOptimizer,
               'ada': tf.train.AdagradOptimizer}[args.optimizer]

  mLow = DynamicModel(args.expdir, learning_rate=args.learning_rate,
                      threads=args.threads, optimizer=optimizer)

  df = LoadData(args.data)
  users = df.groupby('user')
  avg_time = MovingAvg(0.95)

  counter = 0
  for user, grp in users:
    grp = grp.sort_values('date')
    mLow.session.run(mLow.reset_user_embed)

    for i in range(len(grp)):
      row = grp.iloc[i]
      query_len = len(row.query_)

      if query_len < 4:
        continue

      start_time = time.time()
      query = ''.join(row.query_[1:-1])
      result = {'query': query, 'user': row.user, 'idx': i}

      # run the beam search decoding
      if not args.tuning:
        prefix_len = GetPrefixLen(row.user, query, i)
        prefix = row.query_[:prefix_len]
        b = GetCompletions(prefix, 0, mLow, branching_factor=4,
                           beam_size=100)  # always use userid=0
        qlist = [''.join(q.words[1:-1]) for q in reversed(list(b))]
        score = GetRankInList(query, qlist)
        result['score'] = score
        result['top_completion'] = qlist[0]
        result['prefix_len'] = int(prefix_len)

      c, words_in_batch = mLow.Train(row.query_)
      result['length'] = words_in_batch
      result['cost'] = c
      print result
      counter += 1
      t = avg_time.Update(time.time() - start_time)

      if i % 25 == 0:
        sys.stdout.flush()  # flush every so often
        sys.stderr.write('{0}\n'.format(t))
    if counter > args.limit:
        break
