#!/bin/bash


# resolve all dependencies
echo 'Installing Python Dependencies..'
pip install -r requirements.txt

# download nltk data
echo -e "import nltk\nnltk.download('punkt')" | python

# download glove file
echo 'Downloading word2vec (i.e. Glove) embeddings..'
W2V_DIR=./data/word2vec/
mkdir -p ${W2V_DIR}
wget -N http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip -P ${W2V_DIR}
unzip ${W2V_DIR}/glove.840B.300d.zip -d ${W2V_DIR}/
python -m gensim.scripts.glove2word2vec -i ${W2V_DIR}/glove.840B.300d.txt -o ${W2V_DIR}/glove.840B.300d.w2v.txt

