import argparse
import os
import numpy as np
import tensorflow as tf
import sys
from beam import GetCompletions
from dataset import LoadData, Dataset
from helper import GetPrefixLen
from metrics import GetRankInList
from model import MetaModel


parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
parser.add_argument('--data', type=str, action='append', dest='data',
                    help='where to load the data')
parser.add_argument('--learning_rate', type=float, default=None)
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
parser.add_argument('--tuning', action='store_true', dest='tuning',
                    help='when tuning don\'t do beam search decoding',
                    default=False)
args = parser.parse_args()


class DynamicModel(MetaModel):
    
  def __init__(self, expdir, learning_rate=args.learning_rate):
    super(DynamicModel, self).__init__(expdir)

    learning_rate = args.learning_rate
    if learning_rate is None:
      if self.params.use_lowrank_adaptation:
        learning_rate = 0.35
      else:
        learning_rate = 1.1

    self.MakeSession(args.threads)
    self.Restore()
    with self.graph.as_default():
      unk_embed = self.model.user_embed_mat.eval(
        session=self.session)[self.user_vocab['<UNK>']]
      self.reset_user_embed = tf.assign(
          self.model.user_embed_mat, np.expand_dims(unk_embed, 0),
          validate_shape=False)
      self.session.run(self.reset_user_embed)

      self.train_op = tf.no_op()            
      if (self.params.use_lowrank_adaptation or 
          self.params.use_mikolov_adaptation):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.model.avg_loss,
                                           var_list=[self.model.user_embed_mat])
            
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


mLow = DynamicModel(args.expdir)

df = LoadData(args.data)
users = df.groupby('user')

counter = 0
for user, grp in users:
  grp = grp.sort_values('date')
  mLow.session.run(mLow.reset_user_embed)

  for i in range(len(grp)):
    row = grp.iloc[i]
    query_len = len(row.query_)

    if query_len < 4:
      continue

    query = ''.join(row.query_[1:-1])
    result = {'query': query, 'user': row.user, 'idx': i}

    # run the beam search decoding
    if not args.tuning:
      prefix_len = GetPrefixLen(row.user, query)
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

    if i % 15 == 0:
      sys.stdout.flush()  # flush every so often
  if counter > 85540:
      break
