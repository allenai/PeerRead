# PeerReaD
Data and code for ["A Dataset of Peer Reviews (PeerRead): Collection, Insights and NLP Applications"](http://arxiv.org) by Dongyeop Kang, Waleed Ammar, Bhavana Dalvi, Madeleine van Zuylen, Sebastian Kohlmeier, Eduard Hovy and Roy Schwartz, NAACL 2018


# TODOs
* Add a script for downloading the nips papers 2013-2017 (PDFs and reviews), splitting them similarly to the paper, as described [here](./data/nips_2013-2016/README.md)
* Copy arxiv papers from each split in the old repo to one or more subdirectories in `data/arxiv.cs.ai_2007-2017/{train|dev|test}`.
* Add Madeleine's annotations to ICLR 2017 reviews.
* Move code from the old repo to new repo (see agreed upon structure below).


### Structure for PeerRead dataset

```
README.md
requirement.txt
data/
data/conll_2016/
data/nips_2013-2017/
data/arxiv.cs.ai_2007-2017/
data/arxiv.cs.cs_2007-2017/
data/arxiv.cs.lg_2007-2017/
data/iclr_2017/
data/acl_2017/ #only in May 2018
code/
misc/
```

Each section should have a license file, e.g., `data/conll_2016/LICENSE.md`

Each of the splits in a data section should have the following structure for all papers, e.g., for the `data/conll_2016/train` split:
```
data/conll_2016/train/pdfs/$paper_id.pdf
data/conll_2016/train/parsed_pdfs/$paper_id.pdf.json
data/conll_2016/train/reviews/$paper_id.json
```


### How-to-run two NLP tasks: acceptance classification and aspect prediction
Please take a look at code/README.md for detailed instructions.


### Setup Configuration

Running the following script will download the dataset and resolve all dependencies:
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

