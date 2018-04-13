#!/bin/bash
DDIR="../../data/"
DATADIR="$DDIR/iclr_2017" #acl_2017
LABELS=(0 1 2 3 4 5 6 7 8)

TRAIN_TEXT="all" #"review" "paper"
MODEL="dan" #"rnn" "cnn"

for LABEL in "${LABELS[@]}"
do
  echo "Predicting aspect on..." Data=$DATADIR Model=$MODEL Text=$TRAIN_TEXT Label=$LABEL
  python predict.py $DATADIR $TRAIN_TEXT $MODEL $LABEL
  exit
done


