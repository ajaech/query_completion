import json
import os
import random


def GetRandomSetting(force_case=None):
  params = {
    'max_len': 40,
    'iters': 1200000,  # random.randint(550000, 900000),
    'num_units': 600,
    'char_embed_size': 24,
    'user_embed_size': 20,  # random.randint(18, 56),
    'use_layer_norm': True,
    'rank': 60, # random.randint(25, 50),
    'batch_size': 64,
    'dropout': 1.0, # random.choice([0.9, 0.92, 0.94, 0.96, 0.98, 1.0])
    'use_time_features': True,
  }

  if force_case is not None:
    case = force_case
  else:
    case = random.choice(['unadapted', 'mikolov'])
  if case == 'unadapted':
    params['use_mikolov_adaptation'] = False
    params['use_lowrank_adaptation'] = False
  elif case == 'mikolov':
    params['use_mikolov_adaptation'] = True
    params['use_lowrank_adaptation'] = False
  elif case == 'lowrank':
    params['use_mikolov_adaptation'] = False
    params['use_lowrank_adaptation'] = True

  return params


threads = 8
data = """--data /g/ssli/data/LowResourceLM/aol/queries01.train.txt.gz
          --data /g/ssli/data/LowResourceLM/aol/queries02.train.txt.gz 
          --data /g/ssli/data/LowResourceLM/aol/queries03.train.txt.gz
          --data /g/ssli/data/LowResourceLM/aol/queries04.train.txt.gz
          --data /g/ssli/data/LowResourceLM/aol/queries05.train.txt.gz
          --data /g/ssli/data/LowResourceLM/aol/queries06.train.txt.gz""".replace('\n', ' ')

valdata = """
 --valdata /g/ssli/data/LowResourceLM/aol/queries01.dev.txt.gz
 --valdata /g/ssli/data/LowResourceLM/aol/queries02.dev.txt.gz 
 --valdata /g/ssli/data/LowResourceLM/aol/queries03.dev.txt.gz
 --valdata /g/ssli/data/LowResourceLM/aol/queries04.dev.txt.gz
 --valdata /g/ssli/data/LowResourceLM/aol/queries05.dev.txt.gz
 --valdata /g/ssli/data/LowResourceLM/aol/queries06.dev.txt.gz""".replace('\n', ' ')


for i in range(22, 28):
  force = ('mikolov', 'lowrank')[i % 2]
  d = GetRandomSetting(force_case=force)
  fname = os.path.join('settings', '{0}.json'.format(i))
  with open(fname, 'w') as f:
    json.dump(d, f)

    cmd = 'python trainmore.py /n/falcon/s0/ajaech/aolexps/g{0} --threads {1} {2} {3} 2> /n/falcon/s0/ajaech/aolexps/error.g{0}.log'.format(i, threads, data, valdata)
  print cmd
