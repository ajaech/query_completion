"""Holds the Dataset class used for managing training and test data."""
import datetime
import pandas
import numpy as np


def LoadData(filenames, split=True):
  """Load a bunch of files as a pandas dataframe.

  Input files should have three columns for userid, query, and date.
  """
  def Prepare(s):
    s = str(s)
    return ['<S>'] + list(s) + ['</S>']

  dfs = []
  for filename in filenames:
    df = pandas.read_csv(filename, sep='\t', compression='gzip', header=None)
    df.columns = ['user', 'query_', 'date']
    if split:
      df['query_'] = df.query_.apply(Prepare)
    df['user'] = df.user.apply(lambda x: 's' + str(x))

    dates = df.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['hourofday'] = [d.hour for d in dates]
    df['dayofweek'] = [d.dayofweek for d in dates]
    dfs.append(df)
  return pandas.concat(dfs)


class Dataset(object):
    
    def __init__(self, df, char_vocab, user_vocab, batch_size=24, max_len=60):
        self.max_len = max_len
        self.char_vocab = char_vocab
        self.user_vocab = user_vocab
        self.df = df.sample(frac=1)
        self.batch_size = batch_size
        self.current_idx = 0

        self.df['lengths'] = self.df.query_.apply(
          lambda x: min(self.max_len, len(x)))
        
    def GetFeedDict(self, model):
        if self.current_idx + self.batch_size > len(self.df):
            self.current_idx = 0
            
        idx = range(self.current_idx, self.current_idx + self.batch_size)
        self.current_idx += self.batch_size
        
        grp = self.df.iloc[idx]
        
        f1 = np.zeros((self.batch_size, self.max_len))
        user_ids = np.zeros(self.batch_size)
        feed_dict = {
          model.queries: f1,
          model.query_lengths: grp.lengths.values,
          model.user_ids: user_ids,          
        }
        if model.use_time_features:
          feed_dict[model.dayofweek] = grp.dayofweek.values,
          feed_dict[model.hourofday] = grp.hourofday.values
        for i in xrange(len(grp)):
            row = grp.iloc[i]              
            user_ids[i] = self.user_vocab[row.user]
            for j in range(row.lengths):
                f1[i, j] = self.char_vocab[row.query_[j]]
                
        return feed_dict
