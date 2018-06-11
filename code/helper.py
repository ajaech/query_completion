import bunch
import hashlib
import json
import os


def GetPrefixLen(user, query, n=None):
  # choose a random prefix length
  hasher = hashlib.md5()
  hasher.update(user)
  hasher.update(''.join(query))
  if n:
    hasher.update(str(n))
  prefix_len = int(hasher.hexdigest(), 16) % (len(query) - 1)
  prefix_len += 1  # always have at least a single character prefix
  return prefix_len


def GetParams(filename, mode, expdir):
  param_filename = os.path.join(expdir, 'params.json')
  if mode == 'train':
    with open(filename, 'r') as f:
      param_dict = json.load(f)
      params = bunch.Bunch(param_dict)
    with open(param_filename, 'w') as f:
      json.dump(param_dict, f)
  else:
    with open(param_filename, 'r') as f:
      params = bunch.Bunch(json.load(f))
  return params

