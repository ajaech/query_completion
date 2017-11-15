import json
import numpy as np
import os
import random

def GetRandomSetting():
  params = {
    'max_len': 60,
    'iters': random.randint(550000, 900000),
    'num_units': 256,
    'char_embed_size': 16,
    'user_embed_size': random.randint(18, 56),
    'use_layer_norm': True,
    'use_mikolov_adaptation': random.choice([True, False]),
    'use_lowrank_adaptation': random.choice([True, False]),
    'rank': random.randint(20, 40)
  }

  return params


threads = 8

data = """--data /g/ssli/data/LowResourceLM/aol/queries01.train.txt.gz  
          --data /g/ssli/data/LowResourceLM/aol/queries02.train.txt.gz  
          --data /g/ssli/data/LowResourceLM/aol/queries03.train.txt.gz  
          --data /g/ssli/data/LowResourceLM/aol/queries04.train.txt.gz""".replace('\n', ' ')

for i in range(15):
  #d = GetRandomSetting()
  #fname = os.path.join('settings', '{0}.json'.format(i))
  #with open(fname, 'w') as f:
  #  json.dump(d, f)

  cmd = 'python trainmore.py /n/falcon/s0/ajaech/aolexps/a{1} --threads {2} {3} 2> /n/falcon/s0/ajaech/aolexps/error.a{1}.log'.format(None,
    i, threads, data)
  print cmd
