#!/bin/bash

DATADIR="../../data/"
DATAS=("arxiv/cs.cl" "arxiv/cs.lg" "arxiv/cs.ai" "nips/nips_2013" "nips/nips_2014" "nips/nips_2015" "nips/nips_2016" "openreview/ICLR.cc_2017_conference" "conll16" "acl17")

# prepare dataset: split pdfs/reviews into train/dev/test and science-parse them to pdfs/parsed_pdfs/reviews
# under DATADIR, pdfs/reviews directories should be ready in advance
for DATA in "${DATAS[@]}"
do
  echo "preparing dataset: " $DATA
  python prepare.py $DATADIR$DATA
done


