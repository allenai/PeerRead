import os,re,sys,json,logging,itertools,gensim,pprint,functools, random
import numpy as np

logging.getLogger().setLevel(logging.INFO)

def progress(progress, status=""):
    barLength = 15
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Finished.\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [%s] %.2f%% | %s" % ("#"*block + " "*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def pad_sentence(token_list, pad_length, pad_id, reverse=False):
  if reverse:
    token_list = token_list[::-1]
    padding = [pad_id] * (pad_length - len(token_list))
    padded_list = padding + token_list
  else:
    padding = [pad_id] * (pad_length - len(token_list))
    padded_list = token_list + padding
  return padded_list[:pad_length]


def load_embeddings(vocab, load=False):

  glove_embedding = gensim.models.KeyedVectors.load_word2vec_format(load, binary=True)
  embedding_size = len(glove_embedding['the'])

  embedding_var = np.random.normal(0.0, 0.01,[len(vocab), embedding_size] )
  no_embeddings = 0

  for word,wid in vocab.items():
    try:
      embedding_var[wid,:] = glove_embedding[word]
    except KeyError:
      no_embeddings +=1
      continue
  print("num embeddings with no value:{} / {}".format(no_embeddings, len(vocab)))
  return np.array(embedding_var, dtype=np.float32)


def batch_iter(data, batch_size, shuffle=True):
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
  #print 'Total batch, per epoch',data_size, batch_size, num_batches_per_epoch

  if shuffle:
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = data[shuffle_indices]
  else:
    shuffled_data = data

  for batch_num in range(num_batches_per_epoch):
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, data_size)
    yield shuffled_data[start_index:end_index]

