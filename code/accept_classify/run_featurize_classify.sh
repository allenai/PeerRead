#!/bin/bash
#set -x

DATADIR=../../data/iclr_2017
DATASETS=("train" "dev" "test")
FEATDIR=dataset
MAX_VOCAB=30000 #False
ENCODER=w2v
HAND=True



start_time=`date +%s`
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
echo "run-time: $(expr `date +%s` - $start_time) s"


start_time=`date +%s`
echo "Classifying..." $DATASET $ENCODER $MAX_VOCAB $HAND
python classify.py \
	$DATADIR/train/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
	$DATADIR/dev/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
	$DATADIR/test/$FEATDIR/features.svmlite_${MAX_VOCAB}_${ENCODER}_${HAND}.txt \
	$DATADIR/train/$FEATDIR/best_classifer_${MAX_VOCAB}_${ENCODER}_${HAND}.pkl \
	$DATADIR/train/$FEATDIR/features_${MAX_VOCAB}_${ENCODER}_${HAND}.dat
echo "run-time: $(expr `date +%s` - $start_time) s"


