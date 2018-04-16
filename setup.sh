#!/bin/bash

# download glove file
W2V_DIR=./data/word2vec/
mkdir -p ${W2V_DIR}
wget -N http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip -P ${W2V_DIR}
unzip ${W2V_DIR}/glove.840B.300d.zip

# resolve all dependencies
pip install -r requirements.txt
