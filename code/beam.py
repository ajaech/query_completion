"""Helper functions for beam search."""
import numpy as np
from queue import PriorityQueue
from future.utils import implements_iterator


def InitBeam(phrase, user_id, m):
  # Need to find the hidden state for the last char in the prefix.
  prev_hidden = np.zeros((1, 2 * m.params.num_units))
  for word in phrase[:-1]:
    feed_dict = {
       m.model.prev_hidden_state: prev_hidden,
       m.model.prev_word: [m.char_vocab[word]],
       m.model.beam_size: 4
    }
    prev_hidden = m.session.run(m.model.next_hidden_state, feed_dict)

  return prev_hidden


class BeamItem(object):
  """This is a node in the beam search tree.
  
  Each node holds four things: a log probability, a list of previous words, and 
  the two hidden state vectors.
  """
  
  def __init__(self, prev_word, prev_hidden, log_prob=0.0):
    self.log_probs = log_prob
    if type(prev_word) == list:
      self.words = prev_word
    else:
      self.words = [prev_word]
    self.prev_hidden = prev_hidden
    
  def __le__(self, other):
    return self.log_probs <= other.log_probs

  def __lt__(self, other):
    return self.log_probs < other.log_probs

  def __ge__(self, other):
    return self.log_probs >= other.log_probs

  def __gt__(self, other):
    return self.log_probs > other.log_probs

  def __eq__(self, other):
    return self.log_probs == other.log_probs

  def __str__(self):
    return 'beam {0:.3f}: '.format(self.log_probs) + ''.join(self.words)


class BeamQueue(object):
  """Bounded priority queue."""
    
  def __init__(self, max_size=10):
    self.max_size = max_size
    self.size = 0
    self.bound = None
    self.q = PriorityQueue()
        
  def Insert(self, item):
    self.size += 1
    self.q.put((-item.log_probs, item))
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

  def __next__(self):
    if not self.q.empty():
      _, item = self.q.get()
      return item
    raise StopIteration    

  def next(self):
    return self.__next__()


def GetCompletions(prefix, user_id, m, branching_factor=8, beam_size=300, 
                   stop='</S>'):
  """ Find top completions for a given prefix, user and model."""
  m.Lock(user_id)  # pre-compute the adaptive recurrent matrix

  prev_state = InitBeam(prefix, user_id, m)
  nodes = [BeamItem(prefix, prev_state)]

  for i in range(36):
    new_nodes = BeamQueue(max_size=beam_size)
    current_nodes = []
    for node in nodes:
      if i > 0 and node.words[-1] == stop:  # don't extend past the stop token
        new_nodes.Insert(node)  # copy over finished beams
      else:
        current_nodes.append(node)  # these ones will get extended
    if len(current_nodes) == 0:
      return new_nodes  # all beams have finished
    
    # group together all the nodes in the queue for efficient computation
    prev_hidden = np.vstack([item.prev_hidden for item in current_nodes])
    prev_words = np.array([m.char_vocab[item.words[-1]] for item in current_nodes])

    feed_dict = {
      m.model.prev_word: prev_words,
      m.model.prev_hidden_state: prev_hidden,
      m.model.beam_size: branching_factor
    }

    current_char, current_char_p, prev_hidden = m.session.run(
      [m.beam_chars, m.model.selected_p, m.model.next_hidden_state],
      feed_dict)

    for i, node in enumerate(current_nodes):
      for new_word, top_value in zip(current_char[i, :], current_char_p[i, :]):
        new_cost = top_value + node.log_probs
        if new_nodes.CheckBound(new_cost):  # only create a new object if it fits in beam
          new_beam = BeamItem(node.words + [new_word], prev_hidden[i, :],
                              log_prob=new_cost)
          new_nodes.Insert(new_beam)
    nodes = new_nodes
  return nodes


def FirstNonMatch(s1, s2, start=0):
  # returns the position of the first non-matching character
  min_len = min(len(s1), len(s2))
  for i in xrange(start, min_len):
    if s1[i] != s2[i]:
      return i
  return min_len

    
def GetSavedKeystrokes(m, query, branching_factor=4, beam_size=100):
  """Find the shortest prefix that gets the right completion.

  Uses binary search.
  """
  left = 1
  right = len(query)
  while left <= right:
    midpoint = (left + right) / 2
    prefix = ['<S>'] + list(query[:midpoint])
    completions = GetCompletions(
      prefix, 0, m, branching_factor=branching_factor, beam_size=beam_size)
    top_completion = list(completions)[-1]
    top_completion = ''.join(top_completion.words[1:-1])
    if top_completion == query:
      right = midpoint - 1
    else:
      left = midpoint + 1
  return left
