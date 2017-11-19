import argparse
import copy
import hashlib
import os
import pandas
import sys
import numpy as np
import tensorflow as tf
import helper
from beam import BeamItem, BeamQueue
from vocab import Vocab
from model import Model


parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()

expdir = args.expdir


params = helper.GetParams(None, 'eval', args.expdir)


char_vocab = Vocab.Load(os.path.join(expdir, 'char_vocab.pickle'))
user_vocab = Vocab.Load(os.path.join(expdir, 'user_vocab.pickle'))
params.vocab_size = len(char_vocab)
params.user_vocab_size = len(user_vocab)

tf.reset_default_graph()
model = Model(params)

saver = tf.train.Saver(tf.global_variables())
session = tf.Session()
session.run(tf.global_variables_initializer())
saver.restore(session, os.path.join(expdir, 'model.bin'))

class MetaModel(object):
    
    def __init__(self, expdir):
        self.params = helper.GetParams(os.path.join(expdir, 'params.json'),
                                       'eval', expdir)
        self.char_vocab = Vocab.Load(os.path.join(expdir, 'char_vocab.pickle'))
        self.user_vocab = Vocab.Load(os.path.join(expdir, 'user_vocab.pickle'))
        self.params.vocab_size = len(self.char_vocab)
        self.params.user_vocab_size = len(self.user_vocab)
        
        with tf.Graph().as_default():
            self.model = Model(self.params, training_mode=False)
            
            saver = tf.train.Saver(tf.global_variables())
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            saver.restore(self.session, os.path.join(expdir, 'model.bin'))
           
    def Lock(self, user_id):
        self.session.run(self.model.decoder_cell.lock_op,
                         {self.model.user_ids: [user_id]})

    def ScoreQueries(self, top_queries, user_id):
        user_id = self.user_vocab[user_id]
        q = np.zeros((len(top_queries), self.params.max_len))
        q_lengths = np.zeros(len(top_queries))
        user_ids = np.zeros(len(top_queries))
        for i in range(len(top_queries)):
            qq = ['<S>'] + list(top_queries[i]) + ['</S>']
            qq = qq[:60]
            for j in range(len(qq)):
                q[i, j] = self.char_vocab[qq[j]]
            q_lengths[i] = len(qq)
            user_ids[i] = user_id
        feed_dict = {self.model.queries: q, self.model.query_lengths: q_lengths, 
                     self.model.user_ids: user_ids}
        scores = self.session.run(self.model.per_sentence_loss, feed_dict)
        return scores


def InitBeam(phrase, user_id, m):
  prev_c = np.zeros((1, m.params.num_units))
  prev_h = np.zeros((1, m.params.num_units))
  for word in phrase[:-1]:
    feed_dict = {
       m.model.prev_c: prev_c, 
       m.model.prev_h: prev_h,
       m.model.prev_word: [m.char_vocab[word]],
       m.model.beam_size: 4
    }
    prev_c, prev_h = m.session.run(
      [m.model.next_c, m.model.next_h], feed_dict)

  return prev_c, prev_h


def GetCompletions(prefix, user_id, m):
    cell_size = m.params.num_units
    
    m.Lock(user_id)

    starting_phrase = ['<S>'] + list(prefix)
    init_c, init_h = InitBeam(starting_phrase, user_id, m)
    nodes = [BeamItem(starting_phrase, init_c, init_h)]
    total_beam_size = 300
    beam_size = 8

    for i in range(36):
        new_nodes = BeamQueue(max_size=total_beam_size)
        current_nodes = []
        for node in nodes:
            if node.words[-1] == '</S>':  # don't extend past end-of-sentence token
                new_nodes.Insert(node)
            else:
                current_nodes.append(node)
        if len(current_nodes) == 0:
            return new_nodes
        
        prev_c = np.vstack([item.prev_c for item in current_nodes])
        prev_h = np.vstack([item.prev_h for item in current_nodes])
        prev_words = np.array([m.char_vocab[item.words[-1]] for item in current_nodes])
        
        feed_dict = {
            m.model.prev_word: prev_words,
            m.model.prev_c: prev_c,
            m.model.prev_h: prev_h,
            m.model.beam_size: beam_size
        }
        current_word_id, current_word_p, prev_c, prev_h = m.session.run(
            [m.model.selected, m.model.selected_p, m.model.next_c, m.model.next_h],
            feed_dict)
                
        for i, node in enumerate(current_nodes):
            node.prev_c = prev_c[i, :]
            node.prev_h = prev_h[i, :]
            for top_entry, top_value in zip(current_word_id[i, :], current_word_p[i, :]):
                new_word = m.char_vocab[int(top_entry)]
                if new_word != '<UNK>':
                    new_beam = copy.deepcopy(node)
                    new_beam.Update(-np.log(top_value), new_word)
                    new_nodes.Insert(new_beam)
        nodes = new_nodes
    return nodes

df = pandas.read_csv('/g/ssli/data/LowResourceLM/aol/queries01.dev.txt.gz',
                     sep='\t', header=None)
df.columns = ['user', 'query_', 'date']
df['user'] = df.user.apply(lambda x: 's' + str(x))

m = MetaModel(args.expdir)

def GetRankInList(query, qlist):
  if query not in qlist:
    return 0
  return 1.0 / (1.0 + qlist.index(query))


for i in range(9000):
  row = df.iloc[i]
  query_len = len(row.query_)

  if query_len <= 3:
    continue

  # choose a random prefix length
  hasher = hashlib.md5()
  hasher.update(row.user)
  hasher.update(row.query_)
  prefix_len = int(hasher.hexdigest(), 16) % min(query_len - 2, 15)
  prefix_len += 1  # always have at least a single character prefix

  prefix = row.query_[:prefix_len]
  b = GetCompletions(prefix, m.user_vocab[row.user], m)
  qlist = [''.join(q.words[1:-1]) for q in reversed(list(b))]
  score = GetRankInList(row.query_, qlist)
  
  result = {'query': row.query_, 'prefix_len': int(prefix_len),
            'score': score, 'user': row.user}
  print result
  if i % 10 == 0:
    sys.stdout.flush()
