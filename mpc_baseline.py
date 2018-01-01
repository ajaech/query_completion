import numpy as np
import os
import pandas
import pygtrie
import sys
from dataset import LoadData
from helper import GetPrefixLen

import code

query_trie = pygtrie.CharTrie()

dirname = '/g/ssli/data/LowResourceLM/aol'
filenames = ['queries01.train.txt.gz', #'queries02.train.txt.gz',
#             'queries03.train.txt.gz', 'queries04.train.txt.gz', 
#             'queries05.train.txt.gz', 'queries06.train.txt.gz'
]
df = LoadData([os.path.join(dirname, f) for f in filenames], split=False)
z = df.query_.value_counts()
z = z[z > 100]

for q, count in zip(z.index.values, z):
    query_trie[q] = count

cache = {}
def GetTopK(prefix, k=100):
    if prefix in cache:
        return cache[prefix]
    results = query_trie.items(prefix)
    queries, counts = zip(*sorted(results, key=lambda x: -x[-1]))
    cache[prefix] = queries[:k]
    return queries[:k]

df = LoadData(['/g/ssli/data/LowResourceLM/aol/queries01.dev.txt.gz'],
              split=False)

def GetRankInList(query, qlist):
  if query not in qlist:
    return 0
  return 1.0 / (1.0 + qlist.index(query))

for i in range(9000):
  row = df.iloc[i]
  query_len = len(row.query_)

  if query_len <= 3:
    continue
  prefix_len = GetPrefixLen(row.user, row.query_)
  
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
