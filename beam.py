import numpy as np
from Queue import PriorityQueue


def InitBeam(phrase, user_id, m):
  # Need to find the hidden state for the last char in the prefix.
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
  """This is a node in the beam search tree.
  
  Each node holds four things: a log probability, a list of previous words, and 
  the two hidden state vectors.
  """
  
  def __init__(self, prev_word, prev_c, prev_h):
    self.log_probs = 0.0
    if type(prev_word) == list:
      self.words = prev_word
    else:
      self.words = [prev_word]
    self.prev_c = prev_c
    self.prev_h = prev_h

  def __str__(self):
    return 'beam {0:.3f}: '.format(self.Cost()) + ''.join(self.words)

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
    # If the queue is full then we know that there is no chance of a new item
    # being accepted if it's priority is worse than the last thing that got
    # ejected.
    return self.size < self.max_size or self.bound is None or val < self.bound
        
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


def GetCompletions(prefix, user_id, m, branching_factor=8, beam_size=300, 
                   stop='</S>'):
  """ Find top completions for a given prefix, user and model."""
  m.Lock(user_id)  # pre-compute the adaptive recurrent matrix

  init_c, init_h = InitBeam(prefix, user_id, m)
  nodes = [BeamItem(prefix, init_c, init_h)]

  for i in range(36):
    new_nodes = BeamQueue(max_size=beam_size)
    current_nodes = []
    for node in nodes:
      if node.words[-1] == stop:  # don't extend past the stop token
        new_nodes.Insert(node)  # copy over finished beams
      else:
        current_nodes.append(node)  # these ones will get extended
    if len(current_nodes) == 0:
      return new_nodes  # all beams have finished
    
    # group together all the nodes in the queue for efficient computation
    prev_c = np.vstack([item.prev_c for item in current_nodes])
    prev_h = np.vstack([item.prev_h for item in current_nodes])
    prev_words = np.array([m.char_vocab[item.words[-1]] for item in current_nodes])

    feed_dict = {
      m.model.prev_word: prev_words,
      m.model.prev_c: prev_c,
      m.model.prev_h: prev_h,
      m.model.beam_size: branching_factor
    }

    current_char, current_char_p, prev_c, prev_h = m.session.run(
      [m.beam_chars, m.model.selected_p, m.model.next_c, m.model.next_h],
      feed_dict)

    for i, node in enumerate(current_nodes):
      for new_word, top_value in zip(current_char[i, :], current_char_p[i, :]):
        new_cost = top_value + node.log_probs
        if new_nodes.CheckBound(new_cost):  # only create a new object if it fits in beam
          new_beam = BeamItem(list(node.words) + [new_word], prev_c[i, :], prev_h[i, :])
          new_beam.log_probs = new_cost
          new_nodes.Insert(new_beam)
    nodes = new_nodes
  return nodes
