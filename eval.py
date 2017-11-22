import argparse
import bunch
import datetime
import logging
import os
import pandas
import time
import numpy as np
import tensorflow as tf

import helper
from dataset import Dataset, LoadData
from model import Model
from metrics import MovingAvg
from vocab import Vocab


parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
parser.add_argument('--data', type=str, action='append', dest='data',
                    help='where to load the data')
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()

expdir = args.expdir


tf.set_random_seed(int(time.time() * 1000))

params = helper.GetParams(None, 'eval', args.expdir)

char_vocab = Vocab.Load(os.path.join(args.expdir, 'char_vocab.pickle'))
params.vocab_size = len(char_vocab)
user_vocab = Vocab.Load(os.path.join(args.expdir, 'user_vocab.pickle'))
params.user_vocab_size = len(user_vocab)
df = LoadData(args.data)
dataset = Dataset(df, char_vocab, user_vocab, max_len=params.max_len)


model = Model(params, training_mode=False)
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                        intra_op_parallelism_threads=args.threads)
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())
saver.restore(session, os.path.join(expdir, 'model.bin'))


total_word_count = 0
total_log_prob = 0
for idx in range(len(dataset.df) / dataset.batch_size):
  feed_dict = dataset.GetFeedDict(model)
  c, words_in_batch = session.run([model.avg_loss, model.words_in_batch],
                                  feed_dict)
  
  total_word_count += words_in_batch
  total_log_prob += float(c * words_in_batch)
  print '{0}\t{1:.3f}'.format(idx, np.exp(total_log_prob / total_word_count))
