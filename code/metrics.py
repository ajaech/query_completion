import sys

import numpy as np
import tensorflow as tf


class MovingAvg(object):
  
  def __init__(self, p, burn_in=1):
    self.val=None
    self.p=p
    self.burn_in=burn_in

  def Update(self, v):
    if self.burn_in > 0:
      self.burn_in -= 1
      return v

    if self.val is None:
      self.val = v
      return v
    self.val = self.p * self.val + (1.0 - self.p) * v
    return self.val
      

def PrintParams(handle=sys.stdout.write):
  """Print the names of the parameters and their sizes. 

  Args:
    handle: where to write the param sizes to
  """
  handle('NETWORK SIZE REPORT\n')
  param_count = 0
  fmt_str = '{0: <25}\t{1: >12}\t{2: >12,}\n'
  for p in tf.trainable_variables():
    shape = p.get_shape()
    shape_str = 'x'.join([str(x.value) for x in shape])
    handle(fmt_str.format(p.name, shape_str, np.prod(shape).value))
    param_count += np.prod(shape).value
  handle(''.join(['-'] * 60))
  handle('\n')
  handle(fmt_str.format('total', '', param_count))
  if handle==sys.stdout.write:
    sys.stdout.flush()

def GetRankInList(query, qlist):
  # returns the inverse rank of the item in the list
  if query not in qlist:
    return 0
  return 1.0 / (1.0 + qlist.index(query))
