#!/bin/bash

DDIR=../../data
DATADIR=("$DDIR/iclr_2017")
DATASETS=("train" "dev" "test")
FEATDIR=dataset
MAX_VOCAB=False
ENCODER="w2v"
HAND=True

# 1) extract features
for DATASET in "${DATASETS[@]}"
do
  echo "Extracting feautures..." DATA=$DATADIR DATASET=$DATASET ENCODER=$ENCODER ALL_VOCAB=$MAX_VOCAB HAND_FEATURE=$HAND
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

# 2) run baseline model
echo "Classigying..." $DATADIR $DATASET $ENCODER $MAX_VOCAB $HAND
python classify.py \
  $DATADIR/train/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
  $DATADIR/dev/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
  $DATADIR/test/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
  $DATADIR/train/$FEATDIR/best_classifer_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl \
  $DATADIR/train/$FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat
#exit

