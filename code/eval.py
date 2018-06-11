import argparse
import numpy as np

from dataset import Dataset, LoadData
from model import MetaModel
from metrics import MovingAvg


parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
parser.add_argument('--data', type=str, action='append', dest='data',
                    help='where to load the data')
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()
expdir = args.expdir


metamodel = MetaModel(expdir)
model = metamodel.model
metamodel.MakeSessionAndRestore(args.threads)

df = LoadData(args.data)
dataset = Dataset(df, metamodel.char_vocab, metamodel.user_vocab, 
                  max_len=metamodel.params.max_len)

total_word_count = 0
total_log_prob = 0
for idx in range(len(dataset.df) / dataset.batch_size):
  feed_dict = dataset.GetFeedDict(model)
  c, words_in_batch = metamodel.session.run([model.avg_loss, model.words_in_batch],
                                            feed_dict)
  
  total_word_count += words_in_batch
  total_log_prob += float(c * words_in_batch)
  print '{0}\t{1:.3f}'.format(idx, np.exp(total_log_prob / total_word_count))
