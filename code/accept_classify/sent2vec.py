"""
contains different embedding vectorizers and embedding loader
"""
import sys
import numpy as np
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import dill

def import_embeddings(filename="../../data/word2vec/glove.840B.300d.w2v.txt", binary=False):
  """
    Loading pre-trained word embeddings
    For speed-up, you can convert the text file to binary and turn on the switch "binary=True"
  """

  if sys.version_info[0] < 3:
    w2v = gensim.models.Word2Vec.load_word2vec_format(filename, binary=binary)
  else:
    w2v = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=binary)
  return w2v

class MeanEmbeddingVectorizer(object):
  """
   Given a input sentence, output averaged vector of word embeddings in the sentence
  """

  def __init__(self, word2vec,word_counter=False):
    if word_counter:
      self.word2vec = {}
      for w in word_counter:
        if w in word2vec:
          self.word2vec[w] = word2vec[w]
      #import pdb; pdb.set_trace()
    else:
      self.word2vec = word2vec

    # if a text is empty we should return a vector of zeros
    # with the same dimensionality as all the other vectors
    self.dim = len(word2vec['a'])
    print('Dimension: ',self.dim)

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    return np.array([
      np.mean([self.word2vec[w] for w in words if w in self.word2vec]
          or [np.zeros(self.dim)], axis=0)
      for words in X
    ])



class TFIDFEmbeddingVectorizer(object):
  """
   Given a input sentence, output averaged vector of word embeddings weighted by TFIDF scores
  """
  def __init__(self, word2vec,word_counter=False):
    if word_counter:
      self.word2vec = {}
      for w in word2vec:
        if w in word_counter:
          self.word2vec[w] = word2vec[w]
    else:
      self.word2vec = word2vec



    self.word2weight = None
    self.dim = len(word2vec['a'])

  def fit(self, X, y=None):
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    tfidf.fit(X)
    # if a word was never seen - it must be at least as infrequent
    # as any of the known words - so the default idf is the max of
    # known idf's
    max_idf = max(tfidf.idf_)
    self.word2weight = defaultdict(
      lambda: max_idf,
      [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

    return self

  def transform(self, X):
    return np.array([
        np.mean([self.word2vec[w] * self.word2weight[w]
             for w in words if w in self.word2vec] or
            [np.zeros(self.dim)], axis=0)
        for words in X
      ])


