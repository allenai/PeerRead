import sys,os,json, glob,io
from collections import Counter,OrderedDict,defaultdict

sys.path.insert(1,os.path.join(sys.path[0],'..'))
from models.Review import Review
from models.Paper import Paper
from models.ScienceParse import ScienceParse
from models.ScienceParseReader import ScienceParseReader


def get_annots_dic(data_dir='./', input_filename = 'annotation_full.tsv'):
  matched, non_matched = 0, 0
  annots = defaultdict(list)
  with io.open(os.path.join(data_dir,input_filename) ,mode='r',encoding='utf-8') as fin:
    for line in fin:
      tks = line.strip().split('\t')
      if tks[0] == 'id':
        #print 'Keys[%d]'%(len(tks)),tks
        continue
      if len(tks) != 12:
        print('WRONG token length',len(tks), tks)
        continue

      aspect_dic = {}
      aspect_dic['ID'] = tks[0]
      aspect_dic['COMMENTS'] = str(tks[-2])
      aspect_dic['OTHER_KEYS'] = tks[1]

      for idx in range(len(tks)):
        if tks[idx].isdigit():
          tks[idx] = int(tks[idx])
        else:
          tks[idx] = None

      aspect_dic['RECOMMENDATION_UNOFFICIAL'] = tks[9]
      aspect_dic['SUBSTANCE'] = tks[7]
      aspect_dic['APPROPRIATENESS'] = tks[2]
      aspect_dic['MEANINGFUL_COMPARISON'] = tks[6]
      aspect_dic['SOUNDNESS_CORRECTNESS'] = tks[5]
      aspect_dic['ORIGINALITY'] = tks[4]
      aspect_dic['CLARITY'] = tks[3]
      aspect_dic['IMPACT'] = tks[8]
      #aspect_dic['RECOMMENDATION_ORIGINAL'] = float(tks[-1])

      annots[aspect_dic['ID']].append(aspect_dic)
  return annots


def main():

  # Loading annotaions
  annots = get_annots_dic()
  print('Loaded annots: %d papers and %d reviews'%(len(annots), sum([len(v) for k,v in annots.items() ])))

  # Loading reviews, merging them with annotations, and saving into new directory
  data_dir = "../../data/iclr_2017" # args[1]   #train/reviews
  datasets = ['train','dev','test']
  print('Reading reviews from...')
  for dataset in datasets:

    cnt_p, cnt_r = 0, 0

    review_dir = os.path.join(data_dir,  dataset, 'reviews_raw/')
    review_annotated_dir = os.path.join(data_dir,  dataset, 'reviews/')
    scienceparse_dir = os.path.join(data_dir, dataset, 'scienceparse/')
    model_dir = os.path.join(data_dir, dataset, 'model')
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    review_files = sorted(glob.glob('{}/*.json'.format(review_dir)))
    pids = []
    for paper_json_filename in review_files:
      paper = Paper.from_json(paper_json_filename)
      reviews_combined = []
      reviews_annotated = annots[paper.ID]
      reviews_original = paper.REVIEWS

      # overwrite review_annot to reviews in original paper
      reviews_combined += reviews_original

      for r_original in reviews_original:

        r_combined = None #r_original
        for r_annotated in reviews_annotated:

          if r_annotated['OTHER_KEYS'] == r_original.OTHER_KEYS:

            r_combined = r_original
            for k,v in r_annotated.items():
              if k in ['RECOMMENDATION_UNOFFICIAL','SUBSTANCE','APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY','CLARITY','IMPACT']:
                setattr(r_combined, k, v)
            setattr(r_combined, 'IS_ANNOTATED',True)

        if r_combined is None:
          reviews_combined.append(r_original)
        else:
          reviews_combined.append(r_combined)

      paper.REVIEWS = reviews_combined
      cnt_r += len(paper.REVIEWS)


      # save to /reviews_annotated
      json.dump(paper.to_json_object(), open(review_annotated_dir+'/%s.json'%(paper.ID),'w'))
      print(paper.ID, len(paper.REVIEWS))
      cnt_p += 1
    print(dataset, cnt_p, cnt_r)

  # note that we replace reviews/ with reviews_annotated/ to reduce duplicates now

if __name__ == '__main__': main()

