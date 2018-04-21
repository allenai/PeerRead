


### How-to-run

#### (1) Acceptance Classification

To train and test our acceptance classification, please use the following command:
```shell
  cd ./accept_classify/
  ./run_featurize_classify.sh
```

The script "run_featurize_classify.sh" runs feature extraction using "featurize.py" and classification using "classificy.py". The former part takes ~35 minutes and the later takes ~5 seconds on ICLR_2017 dataset.

Here is brief description of each code.
 - "featurize.py" creates (hand-authored and lexical) features for baselines classifiers and save to under dataset folder in each split. This code loads review/paper text and outputs feature vectors in ./{train,dev,test}/dataset/. You should specify type of lexical encoder (e.g., w2v, bow, None) and whether to use hand-authored features or not.
 - "classifiy.py" trains linear classifier using CV and find the best model on dev set. This code loads the featurized vectors from the previous step and outputs accuracies on train/dev/test set.
 - "sent2vec.py" contains different embedding vectorizers and embedding loader.



#### (2) Aspect Prediction

To train and test our aspect predictor, please use the following command:

```shell
  cd ./accept_classify/
  python predict.py "../../data/iclr_2017" {"all","review","paper"} {"dan","rnn","cnn"} {0,1,2,3,4,5,6,7,8}
```


Here is brief description of each code.
 - "pred_models.py" contains three prediction models such as RNN, DAN, CNN
 - "data_helper.py" contains some utility functions for loading data
 - "config.py" contains configurations for each prediction model
 - "predict.py" trains a classifier for predicting review score of each aspect (e.g, recommendation, clarity, etc)
 - "assign_annot_iclr_2017.py" aggregates annotated scores (i.e. annotation_full.tsv) into ICLR_2017 reviews.


### (optional) Data Preparation from your raw data

All of our dataset except NIPS are already preprocessed. For crawling and preprocessing NIPS data, please follow the instruction under ./data/nips_2013-2017/README.md. All other crawlers would be available upon request.

In case you like to crawl the raw dataset and make same data configuration as the paper, please use the following command:

```shell
  python prepare.py ../../data/arxiv/{arxiv.cs.cl_2007-2017,acl_2017,...}
```

Please make sure that pdfs/reviews directories exist and contain raw pdfs/reviews. Note that reviews should be json file of code/model/Review.py class. Then, the script randomly splits them into train/dev/test (0.9/0.05/0.05) into {train/dev/test}/{pdfs/reviews} and science-parse them to {train/dev/test}/{parsed_pdfs}.

Also, download science parser from [here](https://github.com/allenai/science-parse) and locate the science-parse-cli-assembly-1.2.9-SNAPSHOT.jar file under ./code/lib/


