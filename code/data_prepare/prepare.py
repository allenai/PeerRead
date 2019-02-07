import os,sys,glob
from collections import Counter
from shutil import copyfile,rmtree
from sklearn.model_selection import train_test_split

sys.path.insert(1,os.path.join(sys.path[0],'..'))
from models.Paper import Paper
from models.Review import Review

def main(args):
  print(args)
  data_dir = args[1]

  pdf_dir = os.path.join(data_dir, 'pdfs')
  review_dir = os.path.join(data_dir, 'reviews')

  if os.path.exists(os.path.join(data_dir, 'reviews.json')) and not os.path.exists(review_dir):
    print('Loading reviews from a review file')
    papers = Paper.from_softconf_dump(os.path.join(data_dir, 'reviews.json'))
    os.makedirs(review_dir)
    for paper in papers:
      paper.to_json('{}/reviews/{}.json'.format(data_dir,paper.ID))

  if not os.path.exists(pdf_dir) or not os.path.exists(review_dir):
    print('PDF/REVIEW dataset must be ready', pdf_dir, review_dir)

  pdf_files = glob.glob(pdf_dir+'/*.pdf')
  print('Number of pdfs:',len(pdf_files))

  review_files = glob.glob(review_dir+'/*.json')
  print('Number of papers:',len(review_files))
  # checking the decision distributions
  decisions, recs, reviews = [], [], []
  category_dict = {}
  category_types = ['cs.cl','cs.lg','cs.ai']
  for review_file in review_files:
    paper = Paper.from_json(review_file)
    reviews += paper.REVIEWS
    if not paper: continue
    decisions.append(paper.get_accepted())
    if len(paper.get_reviews()) > 0:
      recs.append(paper.get_reviews()[0].get_recommendation())

    # count categories
    matched = False
    categories = paper.SUBJECTS.lower().split(' ')
    for c in categories:
      if c in category_types:
        matched = c
    if matched:
      if matched in category_dict:
        category_dict[matched] += 1
      else:
        category_dict[matched] = 0
    else:
      print(categories, paper.ID)

  print('Paper Decisions:',Counter(decisions))
  print('Review Recommendations:',Counter(recs))
  print('Number of reviews:',len(reviews))
  print('Categories: ', category_dict)

  # science parser
  print('Generating science parses...')
  science_dir = os.path.join(data_dir, 'scienceparse/')
  if not os.path.exists(science_dir):
    print('Parsing papers usnig science-parser...')
    os.makedirs(science_dir)
    os.system('java -Xmx6g -jar ../lib/science-parse-cli-assembly-1.2.9-SNAPSHOT.jar %s -o %s'%(pdf_dir,science_dir))
    science_files = glob.glob(science_dir+'/*.pdf.json')
  else:
    print('Reading parsed science parses...')
    science_files = glob.glob(science_dir+'/*.pdf.json')
  print('Number of science parses:',len(science_files))



  # split to train/dev/test by acceptance
  data_types = ['train','dev','test']
  print('Splitting paper/review/science-parses into train/dev/test')
  split_again = False
  for dtype in data_types:
    data_type_dir = os.path.join(data_dir,dtype)
    if os.path.exists(data_type_dir):
      num_pdfs = len(glob.glob(data_type_dir+'/pdfs'+'/*.pdf'))
      print('file already exists:',data_type_dir, num_pdfs)
    else:
      split_again = True



  if split_again:
    print('splitting ...')
    #os.makedirs(data_type_dir)
    pids = [os.path.basename(pfile).replace('.pdf','') for pfile in pdf_files]
    rids = [os.path.basename(rfile).replace('.json','') for rfile in review_files]
    sids = [os.path.basename(sfile).replace('.pdf.json','') for sfile in science_files]
    ids = []
    for pid in pids:
      if pid in rids and pid in sids:
        ids.append(pid)
    train, validtest = train_test_split(ids, test_size=0.1, random_state = 42)
    dev,test = train_test_split(validtest, test_size=0.5, random_state = 42)

    for didx, data_set in enumerate([train,dev,test]):
      if os.path.exists(os.path.join(data_dir, data_types[didx])):
        rmtree(os.path.join(data_dir, data_types[didx]))
      os.makedirs(os.path.join(data_dir, data_types[didx], 'pdfs'))
      os.makedirs(os.path.join(data_dir, data_types[didx], 'reviews'))
      os.makedirs(os.path.join(data_dir, data_types[didx], 'parsed_pdfs'))
      print('Splitting..',data_types[didx],len(data_set))
      for d in data_set:
        copyfile(
            os.path.join(pdf_dir,d+'.pdf'),
            os.path.join(data_dir, data_types[didx], 'pdfs', d+'.pdf'))

        copyfile(
            os.path.join(review_dir,d+'.json'),
            os.path.join(data_dir, data_types[didx], 'reviews', d+'.json'))

        copyfile(
            os.path.join(science_dir,d+'.pdf.json'),
            os.path.join(data_dir, data_types[didx], 'parsed_pdfs', d+'.pdf.json'))


if __name__ == '__main__': main(sys.argv)
