import argparse
import copy
import hashlib
import os
import pandas
import numpy as np
import tensorflow as tf
import sys
from beam import BeamItem, BeamQueue, InitBeam
from metrics import GetRankInList
from model import MetaModel


parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()


class GenModel(MetaModel):
    
  def __init__(self, expdir):
    super(GenModel, self).__init__(expdir)
    self.MakeSession(args.threads)    
    self.Restore()


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
            new_words = [m.char_vocab[int(x)] for x in current_word_id[i, :]]
            for new_word, top_value in zip(new_words, current_word_p[i, :]):
                if new_word != '<UNK>':
                    new_beam = copy.deepcopy(node)
                    new_beam.Update(-top_value, new_word)
                    new_nodes.Insert(new_beam)
        nodes = new_nodes
    return nodes

df = pandas.read_csv('/g/ssli/data/LowResourceLM/aol/queries01.dev.txt.gz',
                     sep='\t', header=None)
df.columns = ['user', 'query_', 'date']
df['user'] = df.user.apply(lambda x: 's' + str(x))

m = GenModel(args.expdir)


for i in range(9000):
  row = df.iloc[i]
  query_len = len(row.query_)

  if query_len <= 3:
    continue

  # choose a random prefix length
  hasher = hashlib.md5()
  hasher.update(row.user)
  hasher.update(''.join(row.query_))
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
