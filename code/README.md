


### How-to-run

#### (1) Acceptance Classification

To train and test our acceptance classification, please use the following command:
```shell
  cd ./accept_classify/
  DATADIR=../../data/iclr_2017
  DATASETS=("train" "dev" "test")
  FEATDIR=dataset
  MAX_VOCAB=False
  ENCODER=w2v
  HAND=True

  for DATASET in "${DATASETS[@]}"
  do
    echo "Extracting feautures..." DATASET=$DATASET ENCODER=$ENCODER ALL_VOCAB=$MAX_VOCAB HAND_FEATURE=$HAND
    rm -rf $DATADIR/$DATASET/$FEATDIR
    python featurize.py \
      $DATADIR/$DATASET/reviews/ \
      $DATADIR/$DATASET/parsed_pdfs/ \
      $DATADIR/$DATASET/$FEATDIR \
      $DATADIR/train/$FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat \
      $DATADIR/train/$FEATDIR/vect_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl \
      $MAX_VOCAB $ENCODER $HAND
    echo
  done

  echo "Classigying..." $DATASET $ENCODER $MAX_VOCAB $HAND
  python classify.py \
    $DATADIR/train/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
    $DATADIR/dev/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
    $DATADIR/test/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
    $DATADIR/train/$FEATDIR/best_classifer_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl \
    $DATADIR/train/$FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat

```

TODO add some information of each code (featurize.py, classifiy.py, sent2vec.py)
   - create features for baselines classifiers and save to under dataset folder in each split
   - train linear classifier using CV and regularization parameters
   - evaluate the best model on dev set



#### (2) Acceptance Classificatio

To train and test our acceptance classification, please use the following command:

```shell
 cd ./accept_classify/
 python predict.py "../../data/iclr_2017" {"all","review","paper"} {"dan","rnn","cnn"} {0,1,2,3,4,5,6,7,8}
```

TODO add some information of each code (pred_models.py, data_helper.py, config.py, predict.py, assign_annot_iclr_2017.py)
   - predict review scores of each aspect (e.g,, recommendation, clarity, impact, etc)
   - train LSTM-CNN based (0-5] multi-class classifier for each aspect


### (optional) Data Preparation

In case you like to crawl the raw dataset and make same data configuration as the paper, please use the following command:


```shell
  python prepare.py ../../data/arxiv/{arxiv.cs.cl_2007-2017,acl_2017,...}
```

Please make sure that pdfs/reviews directories exist and contain raw pdfs/reviews. Note that reviews should be json file of code/model/Review.py class. Then, the script randomly splits them into train/dev/test (0.9/0.05/0.05) into {train/dev/test}/{pdfs/reviews} and science-parse them to {train/dev/test}/{parsed_pdfs}.

Also, download science parser from [here](https://github.com/allenai/science-parse) and locate the science-parse-cli-assembly-1.2.9-SNAPSHOT.jar file under ./code/lib/


For NIPS crawler, use following crawler:

```shell
 cd ./data_prepare/crawler
 python NIPS_crawl.py
```

All other crawlers would be available upon request.

