import argparse
import copy
import hashlib
import json
import os
import pandas
import numpy as np
import tensorflow as tf
import sys
from dataset import LoadData, Dataset
from model import Model
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
          config = tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                  intra_op_parallelism_threads=args.threads)
          self.session = tf.Session(config=config)
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
            optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
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


mLow = MetaModel(args.expdir)

df = LoadData(args.data)
users = df.groupby('user')

rows = []
for user, grp in users:
    grp = grp.sort_values('date')
    mLow.session.run(mLow.reset_user_embed)
    for i in range(len(grp)):
        row = grp.iloc[i]
        c, words_in_batch = mLow.Train(row.query_, row.user)
        rows.append({'cost': c, 'user': user, 'idx': i, 
                     'length': words_in_batch})
        print rows[-1]
    if len(rows) > 2500:
        break
