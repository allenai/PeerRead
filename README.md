# PeerRead
Data and code for ["A Dataset of Peer Reviews (PeerRead): Collection, Insights and NLP Applications"](http://arxiv.org) by Dongyeop Kang, Waleed Ammar, Bhavana Dalvi, Madeleine van Zuylen, Sebastian Kohlmeier, Eduard Hovy and Roy Schwartz, NAACL 2018

### Structure for PeerRead dataset

```
README.md
requirement.txt
data/
data/conll_2016/
data/nips_2013-2017/
data/arxiv.cs.ai_2007-2017/
data/arxiv.cs.cl_2007-2017/
data/arxiv.cs.lg_2007-2017/
data/iclr_2017/
data/acl_2017/ #only in May 2018
code/
code/data_prepare   # codes for collecting and pre-processing datasets
code/accept_predict # codes for acceptance classification
code/aspect_predict # codes for aspect score prediction
misc/
```

Each section has a license file, e.g., `data/conll_2016/LICENSE.md`

### How-to-run two NLP tasks: acceptance classification and aspect prediction
Please take a look at code/README.md for detailed instructions.


### Setup Configuration

To install dependencies, run:
```
  ./setup.sh
```

This repository has dependencies with:

 * Python 2.7
 * tensorflow 1.2
 * gensim 2.3

### Acknowledgement
 - many codes for collecting accepted papers borrowed from [CanaanShen](https://github.com/CanaanShen/DataProcessor/tree/master/src/Crawler)
 - many codes for aspect prediction borrowed from [jiegzhan](https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn)

