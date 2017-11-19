import pandas
import numpy as np
import os
import copy 
import pygtrie
import hashlib
import sys


query_trie = pygtrie.CharTrie()

dirname = '/g/ssli/data/LowResourceLM/aol'
filenames = ['queries01.train.txt.gz', 'queries02.train.txt.gz',
             'queries03.train.txt.gz', 'queries04.train.txt.gz']
dfs = []
for filename in filenames:
    df = pandas.read_csv(os.path.join(dirname, filename), sep='\t', header=None)
    df.columns = ['user', 'query_', 'date']
    dfs.append(df)
df = pandas.concat(dfs)
z = df.query_.value_counts()

for q, count in zip(z.index.values, z):
    query_trie[q] = count


cache = {}
def GetTopK(prefix, k=300):
    if prefix in cache:
        return cache[prefix]
    results = query_trie.items(prefix)
    queries, counts = zip(*sorted(results, key=lambda x: -x[-1]))
    cache[prefix] = queries[:k]
    return queries[:k]

df = pandas.read_csv('/g/ssli/data/LowResourceLM/aol/queries01.dev.txt.gz',
                     sep='\t', header=None)
df.columns = ['user', 'query_', 'date']
df['user'] = df.user.apply(lambda x: 's' + str(x))

def GetRankInList(query, qlist):
  if query not in qlist:
    return 0
  return 1.0 / (1.0 + qlist.index(query))

for i in range(9000):
  row = df.iloc[i]
  query_len = len(row.query_)

  if query_len <= 3:
    continue

  # choose a random prefix length based on md5 hash
  hasher = hashlib.md5()
  hasher.update(row.user)
  hasher.update(row.query_)
  prefix_len = int(hasher.hexdigest(), 16) % min(query_len - 2, 15)
  prefix_len += 1  # always have at least a single character prefix

  prefix_not_found = False
  prefix = row.query_[:prefix_len]
  if not query_trie.has_subtrie(prefix):
    prefix_not_found = True
    score = 0.0
  else:
    qlist = GetTopK(prefix)
    score = GetRankInList(row.query_, qlist)

  result = {'query': row.query_, 'prefix_len': int(prefix_len),
            'score': score, 'user': row.user, 
            'prefix_not_found': prefix_not_found}
  print result
  if i % 100 == 10:
    sys.stdout.flush()
