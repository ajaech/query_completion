from __future__ import print_function
import argparse
import pandas
import tensorflow as tf
import sys
from beam import GetCompletions
from helper import GetPrefixLen
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
m.MakeSessionAndRestore(args.threads)

for i in range(23000):
  row = df.iloc[i]
  query_len = len(row.query_)

  if query_len <= 3:
    continue

  prefix_len = GetPrefixLen(row.user, row.query_)
  prefix = row.query_[:prefix_len]
  b = GetCompletions(['<S>'] + list(prefix), m.user_vocab[row.user], m,
                     branching_factor=4)
  qlist = [''.join(q.words[1:-1]) for q in reversed(list(b))]
  score = GetRankInList(row.query_, qlist)
  
  result = {'query': row.query_, 'prefix_len': int(prefix_len),
            'score': score, 'user': row.user}
  print(result)
  if i % 10 == 0:
    sys.stdout.flush()
