import sys,os,json, glob

sys.path.insert(1,os.path.join(sys.path[0],'..'))
from models.Review import Review
from models.Paper import Paper
from models.ScienceParse import ScienceParse
from models.ScienceParseReader import ScienceParseReader

def main():

  data_dir = "../../data/iclr_2017" # args[1]   #train/reviews

  datasets = ['train','dev','test']
  aspects = ['RECOMMENDATION', 'SUBSTANCE', 'APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY','CLARITY', 'IMPACT', 'RECOMMENDATION_ORIGINAL']
  annots = json.load(open('annotation_full_structured.json','r'))
  print 'Loaded annots',len(annots)

  # Loading DATA
  data = []
  print 'Reading reviews from...'

  for dataset in datasets:

    cnt_p, cnt_r = 0, 0

    review_dir = os.path.join(data_dir,  dataset, 'reviews/')
    review_annotated_dir = os.path.join(data_dir,  dataset, 'reviews_annotated/')
    scienceparse_dir = os.path.join(data_dir, dataset, 'scienceparse/')
    model_dir = os.path.join(data_dir, dataset, 'model')
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    review_files = sorted(glob.glob('{}/*.json'.format(review_dir)))
    pids = []
    for paper_json_filename in review_files:
      paper = Paper.from_json(paper_json_filename, from_annotated = True)
      reviews = []
      for annot in annots:
        if paper.ID not in annot['ID']: continue
        review = Review(
            RECOMMENDATION = annot['RECOMMENDATION'],
            COMMENTS = annot['COMMENTS'],
            CLARITY = annot['CLARITY'],
            MEANINGFUL_COMPARISON = annot['MEANINGFUL_COMPARISON'],
            SUBSTANCE = annot['SUBSTANCE'],
            SOUNDNESS_CORRECTNESS = annot['SOUNDNESS_CORRECTNESS'],
            APPROPRIATENESS = annot['APPROPRIATENESS'],
            IMPACT = annot['IMPACT'],
            ORIGINALITY = annot['ORIGINALITY'],
            RECOMMENDATION_ORIGINAL = annot['RECOMMENDATION_ORIGINAL']
            )
        reviews.append(review)
      paper.REVIEWS = reviews
      cnt_r += len(reviews)

      # save to /reviews_annotated
      json.dump(paper.to_json_object(), open(review_annotated_dir+'/%s.json'%(paper.ID),'w'))
      print paper.ID, len(reviews)
      cnt_p += 1
    print dataset, cnt_p, cnt_r

if __name__ == '__main__': main()

