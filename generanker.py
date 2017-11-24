import argparse
import hashlib
import os
import pandas
import numpy as np
import tensorflow as tf
import sys
from beam import GetCompletions
from metrics import GetRankInList
from model import MetaModel


parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()


df = pandas.read_csv('/g/ssli/data/LowResourceLM/aol/queries01.dev.txt.gz',
                     sep='\t', header=None)
df.columns = ['user', 'query_', 'date']
df['user'] = df.user.apply(lambda x: 's' + str(x))


m = MetaModel(args.expdir)  # Load the model
m.MakeSession(args.threads)
m.Restore()


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
  b = GetCompletions(['<S>'] + list(prefix), m.user_vocab[row.user], m)
  qlist = [''.join(q.words[1:-1]) for q in reversed(list(b))]
  score = GetRankInList(row.query_, qlist)
  
  result = {'query': row.query_, 'prefix_len': int(prefix_len),
            'score': score, 'user': row.user}
  print result
  if i % 10 == 0:
    sys.stdout.flush()
