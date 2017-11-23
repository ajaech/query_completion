import pandas
import numpy as np


def LoadData(filenames):
  def Prepare(s):
    s = str(s)
    return ['<S>'] + list(s) + ['</S>']

  dfs = []
  for filename in filenames:
    print filename
    df = pandas.read_csv(filename, sep='\t', compression='gzip', header=None)
    df.columns = ['user', 'query_', 'date']
    df['query_'] = df.query_.apply(Prepare)
    df['user'] = df.user.apply(lambda x: 's' + str(x))
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
        
    def GetFeedDict(self, model):
        if self.current_idx + self.batch_size > len(self.df):
            self.current_idx = 0
            
        idx = range(self.current_idx, self.current_idx + self.batch_size)
        self.current_idx += self.batch_size
        
        grp = self.df.iloc[idx]
        
        feeddict = {}
        f1 = np.zeros((self.batch_size, self.max_len))
        len_1 = np.zeros(self.batch_size)
        user_ids = np.zeros(self.batch_size)
        for i in range(len(grp)):
            row = grp.iloc[i]
            len_1[i] = min(self.max_len, len(row.query_))
            user_ids[i] = self.user_vocab[row.user]
            for j in range(int(len_1[i])):
                f1[i, j] = self.char_vocab[row.query_[j]]

                
        return {
            model.queries: f1,
            model.query_lengths: len_1,
            model.user_ids: user_ids
        }
