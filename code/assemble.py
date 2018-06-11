"""
This script is used to create a summary report of key metrics for each 
experiment.
"""
import glob
import json
import numpy as np
import os
import pandas
import re
import code

regex_eval = re.compile(r"'(\w*)': '?([^,]+)'?[,}]")

def FastLoadDynamic(filename):
    rows = []
    with open(filename, 'r') as f:
        for line in f:
            matches = regex_eval.finditer(line)
            d = dict([m.groups() for m in matches])
            if len(d) > 0:
                rows.append(d)
        dynamic_df = pandas.DataFrame(rows)
        if len(dynamic_df) > 0:
            dynamic_df['cost'] = dynamic_df.cost.astype(float)
            dynamic_df['length'] = dynamic_df['length'].astype(float)
            if 'score' in dynamic_df.columns:
              dynamic_df['score'] = dynamic_df['score'].astype(float)
    return dynamic_df

results = []

for dirname in glob.glob('/s0/ajaech/aolexps/c*'):
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
  rank = GetPPL('rank2.txt')
  params['qrank'] = rank
  print rank
  
  filename = os.path.join(dirname, 'dynamic.txt')
  if os.path.exists(filename):
    dyn = FastLoadDynamic(filename)
    if len(dyn) > 0:
      if 'score' in dyn.columns:
        z = np.array(dyn.score.values)
        z[z < 0.1] = 0.0  # crop it at ten
        params['test_mrr'] = 1.0 / max(0.00001, np.mean(dyn.score))
      test_ppl = np.exp((dyn.cost * dyn.length).sum() / dyn.length.sum())
      params['test_ppl'] = test_ppl


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

df.sort_values('qrank').to_csv('results.csv', index=False, sep='\t')
