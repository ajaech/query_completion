import argparse
import copy
import hashlib
import os
import pandas
import numpy as np
import tensorflow as tf
import sys
from beam import BeamItem, BeamQueue, InitBeam
from dataset import LoadData, Dataset
from metrics import GetRankInList
from model import MetaModel, Model
from vocab import Vocab
import helper

import code

parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
parser.add_argument('--data', type=str, action='append', dest='data',
                    help='where to load the data')
parser.add_argument('--learning_rate', type=float, default=0.3)
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()


class DynamicModel(MetaModel):
    
    def __init__(self, expdir, learning_rate=args.learning_rate):
        super(DynamicModel, self).__init__(expdir)
        
        with self.graph.as_default():
          saver = tf.train.Saver(tf.global_variables())
          self.MakeSession(args.threads)
          self.session.run(tf.global_variables_initializer())
          saver.restore(self.session, os.path.join(expdir, 'model.bin'))
            
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
            
    def Train(self, query, user):
        qIds = np.zeros((1, self.params.max_len))
        
        for i in range(min(60, len(query))):
            qIds[0, i] = self.char_vocab[query[i]]
        
        feed_dict = {
            self.model.user_ids: np.array([self.user_vocab[user]]),
            self.model.query_lengths: np.array([len(query)]),
            self.model.queries: qIds
        }
        
        c, words_in_batch, _ = self.session.run(
            [self.model.avg_loss, self.model.words_in_batch, self.train_op], 
            feed_dict)
        return c, words_in_batch


def GetCompletions(prefix, user_id, m):
    cell_size = m.params.num_units
    
    m.Lock()

    init_c, init_h = InitBeam(prefix, user_id, m)
    nodes = [BeamItem(prefix, init_c, init_h)]
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


mLow = DynamicModel(args.expdir)

df = LoadData(args.data)
users = df.groupby('user')

rows = []
for user, grp in users:
    grp = grp.sort_values('date')
    mLow.session.run(mLow.reset_user_embed)

    for i in range(len(grp)):
        row = grp.iloc[i]
        query_len = len(row.query_)

        if query_len < 4:
          continue

        # run the beam search decoding
        # choose a random prefix length
        hasher = hashlib.md5()
        hasher.update(row.user)
        hasher.update(''.join(row.query_))
        prefix_len = int(hasher.hexdigest(), 16) % min(query_len - 2, 15)
        prefix_len += 1  # always have at least a single character prefix

        prefix = row.query_[:prefix_len]
        query = ''.join(row.query_[1:-1])
        b = GetCompletions(prefix, mLow.user_vocab[row.user], mLow)
        qlist = [''.join(q.words[1:-1]) for q in reversed(list(b))]
        score = GetRankInList(query, qlist)

        c, words_in_batch = mLow.Train(row.query_, row.user)
        result = {'query': query, 'prefix_len': int(prefix_len),
                  'score': score, 'user': user, 'idx': i, 
                  'length': words_in_batch, 'cost': c}

        rows.append(result)
        print rows[-1]
        if i % 5 == 0:
          sys.stdout.flush()  # flush every so often
    if len(rows) > 12000:
        break
