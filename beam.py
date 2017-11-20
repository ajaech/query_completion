import collections
import numpy as np
from Queue import PriorityQueue


def InitBeam(phrase, user_id, m):
  prev_c = np.zeros((1, m.params.num_units))
  prev_h = np.zeros((1, m.params.num_units))
  for word in phrase[:-1]:
    feed_dict = {
       m.model.prev_c: prev_c, 
       m.model.prev_h: prev_h,
       m.model.prev_word: [m.char_vocab[word]],
       m.model.beam_size: 4
    }
    prev_c, prev_h = m.session.run(
      [m.model.next_c, m.model.next_h], feed_dict)

  return prev_c, prev_h


class BeamItem(object):
  """This is a node in the beam search tree."""
  
  def __init__(self, prev_word, prev_c, prev_h):
    self.log_probs = 0.0
    if type(prev_word) == list:
      self.words = prev_word
    else:
      self.words = [prev_word]
    self.prev_c = prev_c
    self.prev_h = prev_h

  def __str__(self):
    return 'beam {0:.3f}: '.format(self.Cost()) + ' '.join(self.words)

  def Update(self, log_prob, new_word):
    self.words.append(new_word)
    self.log_probs += log_prob

  def Cost(self):
    return self.log_probs


class BeamQueue(object):
  """Bounded priority queue."""
    
  def __init__(self, max_size=10):
    self.max_size = max_size
    self.size = 0
    self.bound = None
    self.q = PriorityQueue()
        
  def Insert(self, item):
    self.size += 1
    self.q.put((-item.Cost(), item))
    if self.size > self.max_size:
      self.Eject()
            
  def CheckBound(self, val):
    # If the queue is full then we no that there is no chance of a new item
    # being accepted if it's priority is worse than the last thing that got
    # ejected.
    if self.size >= self.max_size and self.bound is not None and val > self.bound:
      return False
    return True
        
  def Eject(self):
    score, _ = self.q.get()
    self.bound = -score
    self.size -= 1
        
  def __iter__(self):
    return self

  def next(self):
    if not self.q.empty():
      _, item = self.q.get()
      return item
    raise StopIteration
