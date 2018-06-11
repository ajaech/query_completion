import argparse
import logging
import os
import pandas
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
parser.add_argument('--valdata', type=str, action='append', dest='valdata',
                    help='where to load validation data', default=[])
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()

expdir = args.expdir


params = helper.GetParams(None, 'eval', args.expdir)

logging.basicConfig(filename=os.path.join(expdir, 'logfile.more.txt'),
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

df = LoadData(args.data)
char_vocab = Vocab.Load(os.path.join(args.expdir, 'char_vocab.pickle'))
params.vocab_size = len(char_vocab)
user_vocab = Vocab.Load(os.path.join(args.expdir, 'user_vocab.pickle'))
params.user_vocab_size = len(user_vocab)
dataset = Dataset(df, char_vocab, user_vocab, max_len=params.max_len)

val_df = LoadData(args.valdata)
valdata = Dataset(val_df, char_vocab, user_vocab, max_len=params.max_len,
                  batch_size=params.batch_size)

model = Model(params, optimizer=tf.train.GradientDescentOptimizer,
              learning_rate=0.05)
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                        intra_op_parallelism_threads=args.threads)
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())
saver.restore(session, os.path.join(expdir, 'model.bin'))

avg_loss = MovingAvg(0.97)
for idx in range(300000):
  feed_dict = dataset.GetFeedDict(model)
  feed_dict[model.dropout_keep_prob] = params.dropout
  c, _ = session.run([model.avg_loss, model.train_op], feed_dict)
  cc = avg_loss.Update(c)
  if idx % 20 == 0 and idx > 0:
    val_c = session.run(model.avg_loss, valdata.GetFeedDict(model))
    logging.info({'iter': idx, 'cost': cc, 'rawcost': c,
                  'valcost': val_c})
  if idx % 999 == 0:
    saver.save(session, os.path.join(expdir, 'model2.bin'),
               write_meta_graph=False)
