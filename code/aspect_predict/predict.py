"""
 predict review scores of each aspect (e.g.,recommendation, clarity, impact, etc)
"""
import sys,os,json, glob,pickle,operator,re,time,logging,shutil,pdb,math
from collections import Counter,OrderedDict,defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import dropwhile
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import cPickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error,accuracy_score

from data_helper import load_embeddings,batch_iter, pad_sentence, progress
from pred_models import CNN,RNN,DAN
from config import CNNConfig, RNNConfig, DANConfig

sys.path.insert(1,os.path.join(sys.path[0],'..'))
from models.Review import Review
from models.Paper import Paper
from models.ScienceParse import ScienceParse
from models.ScienceParseReader import ScienceParseReader



def models(model_text):
  if model_text == 'cnn': return CNN,CNNConfig
  elif model_text == 'rnn': return RNN,RNNConfig
  elif model_text == 'dan': return DAN,DANConfig
  else: return None,None


def preprocess(input, only_char=False, lower=False, stop_remove=False, stemming=False):
  #input = re.sub(r'[^\x00-\x7F]+',' ', input)
  if lower: input = input.lower()
  if only_char:
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)
    input = ' '.join(tokens)
  tokens = word_tokenize(input)
  if stop_remove:
    tokens = [w for w in tokens if not w in stopwords.words('english')]

  # also remove one-length word
  tokens = [w for w in tokens if len(w) > 1]
  return " ".join(tokens)


def evaluate(y, y_):
  return math.sqrt(mean_squared_error(y, y_))


def prepare_data(
    data_dir,
    vocab_path='vocab',
    max_vocab_size = 20000,
    max_len_paper=1000,
    max_len_review=200):


  data_type = data_dir.split('/')[-1]
  vocab_path += '.' + data_type
  if max_vocab_size: vocab_path += '.'+str(max_vocab_size)
  vocab_path = data_dir +'/'+ vocab_path

  label_scale = 5
  if 'iclr' in data_dir.lower():
    fill_missing = False
    aspects = ['RECOMMENDATION', 'SUBSTANCE', 'APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY','CLARITY', 'IMPACT', 'RECOMMENDATION_ORIGINAL']
    review_dir_postfix = '_annotated'
  elif 'acl' in data_dir.lower():
    fill_missing = True
    aspects = ['RECOMMENDATION', 'SUBSTANCE', 'APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY','CLARITY','IMPACT', 'REVIEWER_CONFIDENCE' ]
    review_dir_postfix = ''
  else:
    print 'wrong dataset:',data_dir
    sys.exit(1)


  # Loading datasets
  print 'Reading datasets..'
  datasets = ['train','dev','test']
  paper_content_all = []
  review_content_all = []

  data = defaultdict(list)
  for dataset in datasets:

    review_dir = os.path.join(data_dir,  dataset, 'reviews%s/'%(review_dir_postfix))
    scienceparse_dir = os.path.join(data_dir, dataset, 'parsed_pdfs/')
    model_dir = os.path.join(data_dir, dataset, 'model/')
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_dir)))

    # add all paper/review content to generate corpus for buildinb vocab
    paper_content = []
    review_content = []
    for paper_json_filename in paper_json_filenames:
      d = {}
      paper = Paper.from_json(paper_json_filename)
      paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT, scienceparse_dir)

      review_contents = []
      reviews = []
      for review in paper.REVIEWS:
        review_contents.append(
          preprocess(review.COMMENTS, only_char=False, lower=True, stop_remove=False))
        reviews.append(review)

      d['paper_content'] = preprocess(
          paper.SCIENCEPARSE.get_paper_content(), only_char=False, lower=True, stop_remove=False)
      d['reviews_content'] = review_contents
      d['reviews'] = reviews
      data[dataset].append(d)

  print 'Total number of papers %d' %(np.sum([len(d) for _,d in data.items()]))
  print 'Total number of reviews %d' %(np.sum([len(r['reviews']) for _,d in data.items() for r in d ]))

  # Loading VOCAB
  print 'Building vocab...'
  words =  []
  for _,d in data.items():
    for p in d:
      words += p['paper_content'].split(' ')
      for r in p['reviews_content']:
        words += r.split(' ')
  print "Total words in corpus",len(words)

  vocab = OrderedDict()
  word_counter = Counter(words)
  vocab['PAD'] = 0
  vocab['UNK'] = 1
  for w,c in word_counter.most_common():
    if max_vocab_size:
      if len(vocab) >= max_vocab_size:
        break
    if len(w) and w not in vocab:
      vocab[w] = len(vocab)
  with open(vocab_path, 'w') as fout:
    for w,id in vocab.items():
      fout.write('%s\t%s\n'%(w,id))
  vocab_inv = {int(i):v for v,i in vocab.items()}
  print "Total vocab of size",len(vocab)

  # Loading DATA
  print 'Reading reviews from...'
  data_padded = []
  for dataset in datasets:

    ds = data[dataset]

    x_paper = [] #[None] * len(reviews)
    x_review = [] #[None] * len(reviews)
    y = [] #[None] * len(reviews)


    for d in ds:
      paper_content = d['paper_content']
      reviews_content = d['reviews_content']
      reviews = d['reviews']

      for rid, (review_content, review) in enumerate(zip(reviews_content,reviews)):
        paper_ids = [vocab[w] if w in vocab else 1 for w in paper_content.split(' ') ]
        review_ids = [vocab[w] if w in vocab else 1 for w in review_content.split(' ')]

        paper_ids = pad_sentence(paper_ids, max_len_paper, 0)
        review_ids = pad_sentence(review_ids, max_len_review, 0)

        xone = (paper_ids, review_ids)
        yone = [np.nan] * len(aspects)

        for aid,aspect in enumerate(aspects):
          if aspect in review.__dict__ and review.__dict__[aspect] is not None:
            yone[aid] = float(review.__dict__[aspect])
        #print rid,len(xone[0]), len(xone[1]), yone

        x_paper.append(xone[0])
        x_review.append(xone[1])
        y.append(yone)

    x_paper = np.array(x_paper, dtype=np.int32)
    x_review = np.array(x_review, dtype=np.int32)
    y = np.array(y, dtype=np.float32)

    # add average value of missing aspect value
    if fill_missing:
      col_mean = np.nanmean(y,axis=0)
      inds = np.where(np.isnan(y))
      y[inds] = np.take(col_mean, inds[1])

    print 'Total %s dataset: %d/%d'%(dataset,len(x_paper),len(x_review)),x_paper.shape,x_review.shape, y.shape
    data_padded.append((x_paper,x_review))
    data_padded.append(y)

  return data_padded,vocab,vocab_inv,label_scale,aspects



def choose_label(x,y, size=5, label=False):

  # [size x 9]
  y = np.array(y)

  # (1) only choose label
  if label is not False and label >= 0:
    y = y[:,[label]]

  # (2) remove None/Nan examples
  x = (
    x[0][~np.isnan(y).flatten()],
    x[1][~np.isnan(y).flatten()]
  )
  y = y[~np.isnan(y)]
  y = np.reshape(y, (-1,1))

  assert x[0].shape[0] == y.shape[0]
  assert x[1].shape[0] == y.shape[0]

  mean_aspects = []
  major_aspects = []
  evaluate_mean = []
  evaluate_major = []
  for aid, y_aspect in enumerate(y.T):
    #import pdb; pdb.set_trace()
    mean_aspect = np.average(y_aspect)
    #y_aspect_int = [int(yone) for yone in y_aspect]
    major_aspect = Counter(y_aspect).most_common(1)[0][0]
    mean_aspects.append(mean_aspect)
    major_aspects.append(major_aspect)

    evaluate_mean_aspect = evaluate(y_aspect, [mean_aspect] * len(y_aspect))
    evaluate_major_aspect = evaluate(y_aspect, [major_aspect] * len(y_aspect))
    #print aid,evaluate_mean_aspect, evaluate_major_aspect
    evaluate_mean.append(evaluate_mean_aspect)
    evaluate_major.append(evaluate_major_aspect)

  return x,y, evaluate_mean, evaluate_major, mean_aspects, major_aspects


def main(args):

  argc = len(args)
  data_dir = args[1]   #train/reviews
  train_text = args[2] # 'paper' # paper, review, all
  model_name = args[3] #rnn cnn dan
  label = int(args[4])

  (x_train, y_train, x_dev, y_dev, x_test, y_test),\
      vocab, vocab_inv, label_scale, aspects = \
      prepare_data(
          data_dir,
          max_vocab_size = 35000,
          max_len_paper = 1000,
          max_len_review = 200)

  # choose only given aspect as label among different aspects
  if label >=0:
    aspects = [aspects[label]]
    print 'Labels:',aspects


  # extract only data of interest
  x_train,y_train,evaluate_mean_train,evaluate_major_train,mean_aspects_train,major_aspects_train = \
      choose_label(x_train, y_train, size = label_scale, label=label)
  x_dev,y_dev,evaluate_mean_dev,evaluate_major_dev,_,_ = \
      choose_label(x_dev, y_dev, size = label_scale, label=label)
  x_test,y_test,evaluate_mean_test,evaluate_major_test,_,_ = \
      choose_label(x_test, y_test, size = label_scale, label=label)

  # get mean/major from train on test
  evaluate_mean = []
  evaluate_major = []
  for aid, y_aspect in enumerate(y_test.T):
    mean_aspect = mean_aspects_train[aid]
    major_aspect = major_aspects_train[aid]
    evaluate_mean_aspect = evaluate(y_aspect, [mean_aspect] * len(y_aspect))
    evaluate_major_aspect = evaluate(y_aspect, [major_aspect] * len(y_aspect))
    evaluate_mean.append(evaluate_mean_aspect)
    evaluate_major.append(evaluate_major_aspect)
  print 'Majority (Test)'
  for mean, major,a in zip(evaluate_mean, evaluate_major, aspects):
    print '\t%15s\t%.4f\t%.4f'%(a,mean, major)
  print '\t%15s\t%.4f\t%.4f'%('TOTAL',np.average(evaluate_mean), np.average(evaluate_major))

  # choose train text
  if train_text == 'paper':
    x_train = x_train[0]
    x_dev = x_dev[0]
    x_test = x_test[0]
  elif train_text == 'review':
    x_train = x_train[1]
    x_dev = x_dev[1]
    x_test = x_test[1]
  elif train_text == 'all':
    x_train = np.concatenate(x_train,axis=1)
    x_dev = np.concatenate(x_dev,axis=1)
    x_test = np.concatenate(x_test,axis=1)
  else:
    print 'Wrong'; sys.exit(1)
  max_len = x_train.shape[1]

  print 'x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test))
  print 'y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test))

  timestamp = str(int(time.time()))
  trained_dir = './trained_results/' + timestamp + '/'
  if os.path.exists(trained_dir):
    shutil.rmtree(trained_dir)
  os.makedirs(trained_dir)

  model,config = models(model_name)
  config.seq_length = max_len
  config.vocab_size = len(vocab)
  config.num_classes = len(aspects)


  #load embedding or None
  embedding_mat = load_embeddings(vocab, load="/data/word2vec/glove.840B.300d.w2v.bin") #None

  # loading a model
  model = model(config, embedding = embedding_mat)
  def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

  session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  sess = tf.Session(config=session_conf)
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())
  if embedding_mat is not None:
    sess.run(
        [model.embedding_init],
        feed_dict={model.embedding_placeholder:embedding_mat})


  # Checkpoint files will be saved in this directory during training
  checkpoint_dir = './ckpts/'+ timestamp + '/'
  if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
  os.makedirs(checkpoint_dir)
  checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

  ##############################
  # Training starts here
  ##############################
  best_loss = np.inf
  best_at_step =  0
  for epoch in range(config.num_epochs):

    train_batches = batch_iter(list(zip(x_train, y_train)), config.batch_size)
    train_losses = []
    for train_batch in train_batches:
      x_train_batch, y_train_batch = zip(*train_batch)
      feed_dict = feed_data(x_train_batch, y_train_batch, config.dropout_keep_prob)

      current_step, train_loss,  _  = sess.run(
          [model.global_step,  model.loss, model._train_op], feed_dict)
      train_losses.append(train_loss)

      if current_step % config.print_per_batch==0:
        print '[%d/%d] %.4f'%(epoch,current_step, np.average(train_losses))

      # evaluateuate the model with x_dev and y_dev
      if current_step % config.save_per_batch == 0:
        dev_batches = batch_iter(list(zip(x_dev, y_dev)), config.batch_size)
        dev_losses = []
        aspect_all_ys = {i:[] for i in range(len(aspects))}
        aspect_all_ys_ = {i:[] for i in range(len(aspects))}
        for dev_batch in dev_batches:
          x_dev_batch, y_dev_batch = zip(*dev_batch)
          feed_dict = feed_data(x_dev_batch, y_dev_batch, 1.0)
          dev_loss, dev_logit = sess.run(
              [model.loss, model.logits], feed_dict)
          dev_losses.append(dev_loss)
          #import pdb; pdb.set_trace()
          dev_y = np.array([d for d in dev_batch[:,1]])
          for aid, (y, y_) in enumerate(zip(dev_y.T, dev_logit.T)):
            aspect_all_ys[aid].extend(list(y))
            aspect_all_ys_[aid].extend(list(y_))
        dev_aspect = []
        for aid in range(len(aspects)):
          ys = aspect_all_ys[aid]
          ys_ = aspect_all_ys_[aid]
          dev_aspect.append(evaluate(ys, ys_) )
        #for a,r in zip(aspects, dev_aspect):
        #  print '\t%20s\t%.4f'%(a,r)
        #print '\t%20s\t%.4f'%('TOTAL',np.average(dev_aspect))
        print '[%d] dev loss: %.6f, acc: %.6f'%(current_step,np.average(dev_losses), np.average(dev_aspect))

        # test
        test_batches = batch_iter(list(zip(x_test, y_test)), config.batch_size, shuffle=False)
        aspect_all_ys = {} #[[]] * len(aspects)
        aspect_all_ys_ = {} #[[]] * len(aspects)
        for i in range(len(aspects)):
          aspect_all_ys[i] = []
          aspect_all_ys_[i] = []
        for test_batch in test_batches:
          x_test_batch, y_test_batch = zip(*test_batch)
          feed_dict = feed_data(x_test_batch, y_test_batch, 1.0)
          test_loss, test_logit = sess.run(
              [model.loss, model.logits], feed_dict)
          test_y = np.array([d for d in test_batch[:,1]])
          for aid, (y, y_) in enumerate(zip(test_y.T, test_logit.T)):
            aspect_all_ys[aid].extend(list(y))
            aspect_all_ys_[aid].extend(list(y_))
        test_aspect = []
        for aid in range(len(aspects)):
          ys = aspect_all_ys[aid]
          ys_ = aspect_all_ys_[aid]
          test_aspect.append(evaluate(ys, ys_) )
        print '[%d] test loss: %.4f'%(current_step,np.average(test_aspect))


        if np.average(dev_losses) <= best_loss:
          best_loss, best_at_step = np.average(dev_losses), current_step
          path = saver.save(sess, checkpoint_prefix, global_step=current_step)
          print 'Best loss %.2f at step %d'%(best_loss , best_at_step)
    #print 'Epoch done'
  print 'Training is complete, testing the best model on x_test and y_test'

  print 'Best epoch', best_at_step
  saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))

  test_batches = batch_iter(list(zip(x_test, y_test)), config.batch_size, shuffle=False)
  aspect_all_ys = {} #[[]] * len(aspects)
  aspect_all_ys_ = {} #[[]] * len(aspects)
  for i in range(len(aspects)):
    aspect_all_ys[i] = []
    aspect_all_ys_[i] = []
  for test_batch in test_batches:
    x_test_batch, y_test_batch = zip(*test_batch)
    feed_dict = feed_data(x_test_batch, y_test_batch, 1.0)
    test_loss, test_logit = sess.run(
        [model.loss, model.logits], feed_dict)
    test_y = np.array([d for d in test_batch[:,1]])
    for aid, (y, y_) in enumerate(zip(test_y.T, test_logit.T)):
      aspect_all_ys[aid].extend(list(y))
      aspect_all_ys_[aid].extend(list(y_))
  evaluate_aspect = []
  for aid in range(len(aspects)):
    ys = aspect_all_ys[aid]
    ys_ = aspect_all_ys_[aid]
    evaluate_aspect.append(evaluate(ys, ys_) )
  for a,r in zip(aspects, evaluate_aspect):
    print '\t%20s\t%.4f'%(a,r)
  print '\t%20s\t%.4f'%('TOTAL',np.average(evaluate_aspect))





if __name__ == "__main__": main(sys.argv)

