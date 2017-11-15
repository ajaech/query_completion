import argparse
import pandas
import tensorflow as tf
import bunch
from vocab import Vocab
import random
import numpy as np
from model import Model
import helper
import os
import copy 
import pygtrie

parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()

expdir = args.expdir


params = helper.GetParams(None, 'eval', args.expdir)


query_trie = pygtrie.CharTrie()

df = pandas.read_csv('/g/ssli/data/LowResourceLM/aol/queries01.train.txt.gz',
                     sep='\t', header=None)
df.columns = ['user', 'query_', 'date']
z = df.query_.value_counts()
for q, count in zip(z.index.values, z):
    query_trie[q] = count


cache = {}
def GetTopK(prefix, k=20):
    if prefix in cache:
        return cache[prefix]
    results = query_trie.items(prefix)
    queries, counts = zip(*sorted(results, key=lambda x: -x[-1]))
    cache[prefix] = queries[:k]
    return queries[:k]


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

def GetRankedQueries(user_id, prefix):
    top_queries = GetTopK(prefix, 50)
    q = np.zeros((len(top_queries), params.max_len))
    q_lengths = np.zeros(len(top_queries))
    user_ids = np.zeros(len(top_queries))
    for i in range(len(top_queries)):
        qq = ['<S>'] + list(top_queries[i]) + ['</S>']
        qq = qq[:60]
        for j in range(len(qq)):
            q[i, j] = char_vocab[qq[j]]
        q_lengths[i] = len(qq)
        user_ids[i] = user_vocab[user_id]
    feed_dict = {model.queries: q, model.query_lengths: q_lengths, model.user_ids: user_ids}
    scores = session.run(model.per_sentence_loss, feed_dict)
    ranks = np.argsort(scores)
    return [scores[i] for i in ranks], [top_queries[i] for i in ranks]


scores = []
for i in range(2000):
    row = df.iloc[i]
    
    a, b = GetRankedQueries(user_id=user_vocab[row.user], prefix=row.query_[:2])
    if row.query_ in b:
        scores.append(1.0 / (1.0 + b.index(row.query_)))
    else:
        scores.append(0.0)

    if np.mean(scores) > 0.0:
      print i, 1.0 / np.mean(scores)
