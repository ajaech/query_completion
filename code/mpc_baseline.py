from __future__ import print_function
import gzip
import os
import pandas
import pygtrie
import re
import sys
import numpy as np
from dataset import LoadData
from helper import GetPrefixLen


query_trie = pygtrie.CharTrie()

dirname = '../data'
filenames = ['queries01.train.txt.gz', 'queries02.train.txt.gz',
             'queries03.train.txt.gz', 'queries04.train.txt.gz', 
             'queries05.train.txt.gz', 'queries06.train.txt.gz'
]
df = LoadData([os.path.join(dirname, f) for f in filenames], split=False)
z = df.query_.value_counts()
z = z[z > 2]

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


def GetRankInList(query, qlist):
  if query not in qlist:
    return 0
  return 1.0 / (1.0 + qlist.index(query))


regex_eval = re.compile(r"'(\w*)': '?([^,]+)'?[,}]")

def FastLoadDynamic(filename):
    rows = []
    with gzip.open(filename, 'r') as f:
      for line in f:        
        if type(line) != str:
          line = line.decode('utf8')  # for python3 compatibility
        matches = regex_eval.finditer(line)
        d = dict([m.groups() for m in matches])
        if len(d) > 0:
          rows.append(d)
        else:
          print('bad line')
          print(line)
    dynamic_df = pandas.DataFrame(rows)
    if len(dynamic_df) > 0:
        if 'cost' in dynamic_df.columns:
            dynamic_df['cost'] = dynamic_df.cost.astype(float)
        if 'length' in dynamic_df.columns:
            dynamic_df['length'] = dynamic_df['length'].astype(float)
        dynamic_df['score'] = dynamic_df['score'].astype(float)
    return dynamic_df

rank_data = FastLoadDynamic('../data/predictions.log.gz')

for i in range(len(rank_data)):
    row = rank_data.iloc[i]
    if sys.version_info.major == 2:
      query = row['query'][:-1].decode('string_escape')
    else:  #python3 compatability
      query = row['query'][:-1].encode('utf8').decode('unicode_escape')
    query_len = len(query)

    prefix_len = int(row.prefix_len)

    prefix_not_found = False
    prefix = query[:prefix_len]
    if not query_trie.has_subtrie(prefix):
      prefix_not_found = True
      score = 0.0
    else:
      qlist = GetTopK(prefix)
      score = GetRankInList(query, qlist)

    result = {'query': query, 'prefix_len': int(prefix_len),
              'score': score, 'user': row.user, 
              'prefix_not_found': prefix_not_found}
    print(result)
    if i % 100 == 0:
      sys.stdout.flush()
