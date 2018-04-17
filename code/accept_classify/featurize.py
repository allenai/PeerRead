"""
create (hand-authored and lexical) features for baselines classifiers and save to under dataset folder in each split
"""

import sys,os,random,json,glob,operator,re
import cPickle as pkl
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from itertools import dropwhile
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

sys.path.insert(1,os.path.join(sys.path[0],'..'))
from models.Review import Review
from models.Paper import Paper
from models.ScienceParse import ScienceParse
from models.ScienceParseReader import ScienceParseReader

def read_features(ifile):
  idToFeature = dict()
  with open(ifile,"rb") as ifh:
    for l in ifh:
      e = l.rstrip().decode("utf-8").split("\t")
      if len(e) == 2:
        idToFeature[e[1]] = e[0]
  return idToFeature

def save_features_to_file(idToFeature, feature_output_file):
  with open(feature_output_file, 'wb') as ofh:
    sorted_items = sorted(idToFeature.items(), key=operator.itemgetter(1))
    for i in sorted_items:
      str = "{}\t{}\n".format(i[1],i[0]).encode("utf-8")
      ofh.write(str)

def save_vect(vect, ofile):
  pkl.dump(vect, open(ofile, "wb"))

def load_vect(ifile):
  return pkl.load( open( ifile, "rb" ) )

def count_words(corpus, HFW_proportion, most_frequent_words_proportion, ignore_infrequent_words_thr):
  counter = Counter(corpus)
  most_common = [x[0] for x in counter.most_common(int(len(counter)*HFW_proportion))]
  most_common2 = [x[0] for x in counter.most_common(int(len(counter) * (HFW_proportion+most_frequent_words_proportion)))]

  most_frequent_words = set()
  least_frequent_words = set()

  for w in counter:
    if w in most_common2 and w not in most_common:
      most_frequent_words.add(w)
    elif counter[w] < ignore_infrequent_words_thr:
      least_frequent_words.add(w)
  return most_common, most_frequent_words, least_frequent_words


def preprocess(input, only_char=False, lower=False, stop_remove=False, stemming=False):
  input = re.sub(r'[^\x00-\x7F]+',' ', input)
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


def main(args, lower=True, max_vocab_size = False, encoder='bowtfidf'):
  argc = len(args)

  if argc < 9:
    print("Usage:", args[0], "<paper-json-dir> <scienceparse-dir> <out-dir> <submission-year> <feature output file> <tfidf vector file> <max_vocab_size> <encoder> <hand-feature>")
    return -1

  paper_json_dir = args[1]      #train/reviews
  scienceparse_dir = args[2]    #train/parsed_pdfs
  out_dir = args[3]             #train/dataset
  feature_output_file = args[4] #train/dataset/features.dat
  vect_file = args[5]           #train/dataset/vect.pkl
  max_vocab_size = False if args[6]=='False' else int(args[6])  # False or integer
  encoder = False if args[7]=='False' else str(args[7])
  hand = False if args[8]=='False' else str(args[8])


  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  is_train = True
  vect = None
  idToFeature = None
  if os.path.isfile(feature_output_file):
    is_train = False
    idToFeature = read_features(feature_output_file)
    if encoder:
      print 'Loading vector file from...',vect_file
      vect = load_vect(vect_file)
  else:
    print 'Loading vector file from scratch..'
    idToFeature = dict()

  outLabelsFile = open(out_dir + '/labels_%s_%s_%s.tsv'%(str(max_vocab_size), str(encoder),str(hand)), 'w')
  outIDFile = open(out_dir + '/ids_%s_%s_%s.tsv'%(str(max_vocab_size), str(encoder),str(hand)), 'w')
  outSvmLiteFile = open(out_dir + '/features.svmlite_%s_%s_%s.txt'%(str(max_vocab_size), str(encoder),str(hand)), 'w')




  ################################
  # read reviews
  ################################
  print 'Reading reviews from...',paper_json_dir
  paper_content_corpus = [] #""
  paper_json_filenames = sorted(glob.glob('{}/*.json'.format(paper_json_dir)))
  papers = []
  for paper_json_filename in paper_json_filenames:
    paper = Paper.from_json(paper_json_filename)
    paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT, scienceparse_dir)
    paper_content_corpus.append(paper.SCIENCEPARSE.get_paper_content())
    papers.append(paper)
  random.shuffle(papers)
  print 'Total number of reviews',len(papers)


  def get_feature_id(feature):
    if feature in idToFeature:
      return idToFeature[feature]
    else:
      return None

  def addFeatureToDict(fname):
    id = len(idToFeature)
    idToFeature[fname] = id


  ################################
  # Initialize vocabularty
  ################################
  outCorpusFilename =  out_dir + '/corpus.pkl'
  if not os.path.isfile(outCorpusFilename):
    paper_content_corpus = [preprocess(p, only_char=True, lower=True, stop_remove=True) for p in paper_content_corpus]
    paper_content_corpus_words =  []
    for p in paper_content_corpus:
      paper_content_corpus_words += p.split(' ')
    pkl.dump(paper_content_corpus_words, open(outCorpusFilename, 'wb'))
  else:
    paper_content_corpus_words = pkl.load(open(outCorpusFilename,'rb'))
  print 'Total words in corpus',len(paper_content_corpus_words)





  ################################
  # Encoding
  ################################
  print 'Encoding..',encoder
  # 1) tf-idf features on title/author_names/domains
  if not encoder:
    print 'No encoder',encoder
  elif encoder in ['bow', 'bowtfidf']:
    word_counter = Counter(paper_content_corpus_words)
    # vocab limit by frequency
    if max_vocab_size:
      word_counter = dict(word_counter.most_common()[:max_vocab_size])
    vocabulary = dict()
    for w in word_counter:
      if len(w) and w not in vocabulary:
        if is_train:
          vocabulary[w] = len(vocabulary)
          addFeatureToDict(w)
        else:
          fid = get_feature_id(w)
          if fid is not None:
            vocabulary[w] = fid
    print("Got vocab of size",len(vocabulary))
    if is_train:
      print 'Saving vectorized',vect_file
      if encoder == 'bow':
        vect = CountVectorizer( max_df=0.5, analyzer='word', stop_words='english', vocabulary=vocabulary)
      else:
        vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word',
                     stop_words='english', vocabulary=vocabulary)
      vect.fit([p for p in paper_content_corpus])
      save_vect(vect, vect_file)

  # 2) sentence encoder features
  elif encoder in ['w2v', 'w2vtfidf']:
    from sent2vec import MeanEmbeddingVectorizer,TFIDFEmbeddingVectorizer,import_embeddings
    if is_train:
      w2v = import_embeddings()
      vect = MeanEmbeddingVectorizer(w2v) if encoder=='w2v' else TFIDFEmbeddingVectorizer(w2v)
      for f in range(vect.dim):
        #fid = get_feature_id()
        addFeatureToDict('%s%d'%(encoder,f))
      print 'Saving vectorized',vect_file
      if encoder == 'w2vtfidf':
        vect.fit([p for p in paper_content_corpus])
      save_vect(vect, vect_file)
  else:
    print 'Wrong type of encoder',encoder
    sys.exit(1)


  ################################
  # Add features
  ################################
  if encoder:
    all_titles = []
    for p in papers:
      sp = p.get_scienceparse()
      title = p.get_title()
      all_title = preprocess(title, only_char=True, lower=True, stop_remove=True)
      all_titles.append(all_title)
    all_titles_features = vect.transform(all_titles)

  if is_train:
    print 'saving features to file',feature_output_file
    if hand:
      addFeatureToDict("get_most_recent_reference_year")
      addFeatureToDict("get_num_references")
      addFeatureToDict("get_num_refmentions")
      addFeatureToDict("get_avg_length_reference_mention_contexts")
      addFeatureToDict("abstract_contains_deep")
      addFeatureToDict("abstract_contains_neural")
      addFeatureToDict("abstract_contains_embedding")
      addFeatureToDict("abstract_contains_outperform")
      addFeatureToDict("abstract_contains_novel")
      addFeatureToDict("abstract_contains_state_of_the_art")
      addFeatureToDict("abstract_contains_state-of-the-art")

      addFeatureToDict("get_num_recent_references")
      addFeatureToDict("get_num_ref_to_figures")
      addFeatureToDict("get_num_ref_to_tables")
      addFeatureToDict("get_num_ref_to_sections")
      addFeatureToDict("get_num_uniq_words")
      addFeatureToDict("get_num_sections")
      addFeatureToDict("get_avg_sentence_length")
      addFeatureToDict("get_contains_appendix")
      addFeatureToDict("proportion_of_frequent_words")
      addFeatureToDict("get_title_length")
      addFeatureToDict("get_num_authors")

      addFeatureToDict("get_num_ref_to_equations")
      addFeatureToDict("get_num_ref_to_theorems")


    save_features_to_file(idToFeature, feature_output_file)

  id = 1
  hfws, most_frequent_words, least_frequent_words = count_words(paper_content_corpus_words, 0.01, 0.05, 3)
  for p in papers:
    outIDFile.write(str(id) + "\t" + str(p.get_title()) + "\n")
    rs = [r.get_recommendation() for r in p.get_reviews()]
    rec = int(p.get_accepted() == True)
    outLabelsFile.write(str(rec))
    outSvmLiteFile.write(str(rec) + " ")

    sp = p.get_scienceparse()

    if encoder:
      title_tfidf = all_titles_features[id-1]
      if encoder.startswith('bow'):
        nz = title_tfidf.nonzero()[1]
        for word_id in sorted(nz):
          outSvmLiteFile.write(str(word_id)+":"+ str(title_tfidf[0,word_id])+" ")
      elif encoder.startswith('w2v'):
        for word_id in range(vect.dim):
          outSvmLiteFile.write(str(word_id)+":"+ str(title_tfidf[word_id])+" ")
      else:
        print 'wrong ecndoer', encoder
        sys.exit(1)

    if hand:
      outSvmLiteFile.write(
          str(get_feature_id("get_most_recent_reference_year")) + ":" +
          str(sp.get_most_recent_reference_year()-2000) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_num_references")) + ":" +
          str(sp.get_num_references()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_num_refmentions")) + ":" +
          str(sp.get_num_refmentions()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_avg_length_reference_mention_contexts")) + ":" +
          str(sp.get_avg_length_reference_mention_contexts()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("abstract_contains_deep")) + ":" +
          str(int(p.abstract_contains_a_term("deep"))) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("abstract_contains_neural")) + ":" +
          str(int(p.abstract_contains_a_term("neural"))) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("abstract_contains_embedding")) + ":" +
          str(int(p.abstract_contains_a_term("embedding"))) + " ")

      outSvmLiteFile.write(
          str(get_feature_id("abstract_contains_outperform")) + ":" +
          str(int(p.abstract_contains_a_term("outperform"))) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("abstract_contains_novel")) + ":" +
          str(int(p.abstract_contains_a_term("novel"))) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("abstract_contains_state_of_the_art")) + ":" +
          str(int(p.abstract_contains_a_term("state of the art"))) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("abstract_contains_state-of-the-art")) + ":" +
          str(int(p.abstract_contains_a_term("state-of-the-art"))) + " ")

      outSvmLiteFile.write(
          str(get_feature_id("get_num_recent_references")) + ":" +
          str(sp.get_num_recent_references(2017)) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_num_ref_to_figures")) + ":" +
          str(sp.get_num_ref_to_figures()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_num_ref_to_tables")) + ":" +
          str(sp.get_num_ref_to_tables()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_num_ref_to_sections")) + ":" +
          str(sp.get_num_ref_to_sections()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_num_uniq_words")) + ":" +
          str(sp.get_num_uniq_words()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_num_sections")) + ":" +
          str(sp.get_num_sections()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_avg_sentence_length")) + ":" +
          str(sp.get_avg_sentence_length()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_contains_appendix")) + ":" +
          str(sp.get_contains_appendix()) + " ")

      outSvmLiteFile.write(
          str(get_feature_id("proportion_of_frequent_words")) + ":" +
          str(round(sp.get_frequent_words_proportion(hfws, most_frequent_words, least_frequent_words),3)) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_title_length")) + ":" +
          str(p.get_title_len()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_num_authors")) + ":" +
          str(sp.get_num_authors()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_num_ref_to_equations")) + ":" +
          str(sp.get_num_ref_to_equations()) + " ")
      outSvmLiteFile.write(
          str(get_feature_id("get_num_ref_to_theorems")) + ":" +
          str(sp.get_num_ref_to_theorems()) + " ")

    outSvmLiteFile.write("\n")
    id += 1

  outLabelsFile.close()
  outIDFile.close()
  outSvmLiteFile.close()
  print 'saved',outLabelsFile.name
  print 'saved',outIDFile.name
  print 'saved',outSvmLiteFile.name


if __name__ == "__main__":
  main(sys.argv)




