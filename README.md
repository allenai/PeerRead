# PeerReaD
Data and code for ["A Dataset of Peer Reviews (PeerRead): Collection, Insights and NLP Applications"](http://arxiv.org) by Dongyeop Kang, Waleed Ammar, Bhavana Dalvi, Madeleine van Zuylen, Sebastian Kohlmeier, Eduard Hovy and Roy Schwartz, NAACL 2018

## The PeerRead dataset
PearRead is a dataset of scientific peer reviews available to help researchers study this important artifact.
The dataset consists of over 14K paper drafts and the corresponding accept/reject decisions in top-tier venues including ACL, NIPS and ICLR, as well as over 10K textual peer reviews written by experts for a subset of the papers.

We structured the dataset into sections each corresponding to a venue or an arxiv category, e.g., [./data/acl_2017](./data/acl_2017) and [./data/arxiv.cs.cl_2007-2017](./data/arxiv.cs.cl_2007-2017). Each section is further split into the train/dev/test splits (same splits used in the paper). Due to licensing constraints, we provide instructions for downloading the data for some sections instead of including it in this repository, e.g., [./data/nips_2013-2017/README.md](./data/nips_2013-2017/README.md).

## Models
In order to experiment with (and hopefully improve) our models for aspect prediction and for predicting whether a paper will be accepted, see [./code/README.md](./code/README.md).

## Setup Configuration
Run `./setup.sh` at the root of this repository to install dependencies and download some of the larger data files not included in this repo.

## Acknowledgement
 - We use some of the code in [CanaanShen](https://github.com/CanaanShen/DataProcessor/tree/master/src/Crawler) for web crawling.
 - We use some of the code in [jiegzhan](https://github.com/jiegzhan/multi-class-text-classification-cnn-rnn) for our aspect prediction experiments.
- This work would not have been possible without the efforts of Rich Gerber and Paolo Gai (developers of the [softconf.com](softconf.com) conference management system), Stefan Riezler, Yoav Goldberg (chairs of CoNLL 2016), Min-Yen Kan, Regina Barzilay (chairs of ACL 2017) for allowing authors and reviewers to opt-in for this dataset during the official review process.
- We thank the [openreview.net](openreview.net), [arxiv.org](arxiv.org) and [semanticscholar.org](semanticscholar.org) teams for their commitment to promoting transparency and openness in scientific communication. 
