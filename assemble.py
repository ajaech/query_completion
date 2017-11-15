import glob
import json
import numpy as np
import os
import pandas
import code

results = []

for dirname in glob.glob('/s0/ajaech/aolexps/*'):
  if os.path.isfile(dirname):
    continue

  # load the params file
  params_filename = os.path.join(dirname, 'params.json')
  if not os.path.exists(params_filename):
    continue  # this is just an empty directory
  with open(params_filename, 'r') as g:
    params = json.load(g)
  params['dir'] = dirname

  model_filename = os.path.join(dirname, 'model.bin.index')
  if os.path.exists(model_filename):
    modtime = os.path.getmtime(model_filename)
    params['finish_time'] = modtime

  def GetPPL(name):
    filename = os.path.join(dirname, name)
    if os.path.exists(filename):
      with open(filename, 'r') as f:
        lines = f.readlines()
      if len(lines):
        fields = lines[-1].split()
        if len(fields):
          try:
            ppl = float(fields[-1])
            return ppl
          except:
            return None
    return None

  ppl = GetPPL('ppl.txt')
  print ppl, dirname
  params['ppl'] = ppl
  ppl = GetPPL('pplfinal.txt')
  params['pplfinal'] = ppl
  rank = GetPPL('rank.txt')
  params['qrank'] = rank
  rank2 = GetPPL('rank2.txt')
  params['rank2'] = rank2
  results.append(params)


df = pandas.DataFrame(results)
if 'acc' in df.columns:
  df = df.sort_values('acc')
else:
  df = df.sort_values('ppl')

# delete boring columns
for column in df.columns:
  if df[column].dtype == list:
    continue
  if len(df[column].unique()) == 1:
    del df[column]

df.to_csv('results.csv', index=False, sep='\t')
