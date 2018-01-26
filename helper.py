import bunch
import hashlib
import json
import os
import numpy as np


def GetPrefixLen(user, query, n=None):
  # choose a random prefix length
  hasher = hashlib.md5()
  hasher.update(user)
  hasher.update(''.join(query))
  if n:
    hasher.update(str(n))
  prefix_len = int(hasher.hexdigest(), 16) % min(len(query), 15)
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


def levenshtein(source, target, cutoff=None):
    if len(source) < len(target):
        return levenshtein(target, source, cutoff=cutoff)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

        if cutoff is not None and previous_row.min() > cutoff:
          return cutoff

    return previous_row[-1]
