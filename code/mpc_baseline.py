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

test_data = ['queries01.dev.txt.gz']
#test_data = ['queries08.train.txt.gz', 'queries08.dev.txt.gz', 
#             'queries08.test.txt.gz']
df = LoadData([os.path.join(dirname, f) for f in test_data],
              split=False)
users = df.groupby('user')

def GetRankInList(query, qlist):
  if query not in qlist:
    return 0
  return 1.0 / (1.0 + qlist.index(query))

import re

regex_eval = re.compile(r"'(\w*)': '?([^,]+)'?[,}]")

def FastLoadDynamic(filename):
    rows = []
    with open(filename, 'r') as f:
        for line in f:
            matches = regex_eval.finditer(line)
            d = dict([m.groups() for m in matches])
            if len(d) > 0:
                rows.append(d)
            else:
                print 'bad line'
                print line
        dynamic_df = pandas.DataFrame(rows)
        if len(dynamic_df) > 0:
            if 'cost' in dynamic_df.columns:
                dynamic_df['cost'] = dynamic_df.cost.astype(float)
            if 'length' in dynamic_df.columns:
                dynamic_df['length'] = dynamic_df['length'].astype(float)
            dynamic_df['score'] = dynamic_df['score'].astype(float)
        return dynamic_df

rank_data = FastLoadDynamic('/n/falcon/s0/ajaech/aolexps/g23/7dynamic.txt')

for i in range(len(rank_data)):
    row = rank_data.iloc[i]
    query = row['query'][:-1].decode('string_escape')
    query_len = len(query)

    # offset by one because of missing <S> token
    prefix_len = int(row.prefix_len) - 1

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
    print result
    if i % 100 == 0:
      sys.stdout.flush()

