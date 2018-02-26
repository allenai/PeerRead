import re,io
import json
import sys
from Review import Review

class Paper:
  """A paper class, which contains relevant fields and a list of reviews"""
  def __init__(self, TITLE, ABSTRACT, ID, REVIEWS, AUTHORS=None, CONFERENCE=None, ACCEPTED=None, SCORE=None,
         PUBLICATION_TYPE=None, SCIENCEPARSE=None, KEYWORDS=None, AUTHOR_EMAILS=None, DATE_OF_SUBMISSION=None,
         SUBJECTS=None,COMMENTS=None,VERSION=None,HISTORIES=None):
    self.TITLE = TITLE
    self.ABSTRACT = re.sub("\\n", " ", ABSTRACT)
    self.ID = ID
    self.AUTHORS = AUTHORS
    self.REVIEWS = REVIEWS
    self.SCIENCEPARSE = SCIENCEPARSE
    self.CONFERENCE = CONFERENCE
    self.ACCEPTED = ACCEPTED
    self.SCORE = SCORE
    self.PUBLICATION_TYPE = PUBLICATION_TYPE
    self.KEYWORDS = KEYWORDS
    self.AUTHOR_EMAILS = AUTHOR_EMAILS
    self.DATE_OF_SUBMISSION = DATE_OF_SUBMISSION

    # additional properties for arxiv papers
    self.SUBJECTS = SUBJECTS
    self.COMMENTS = COMMENTS
    self.VERSION = VERSION
    self.HISTORIES = HISTORIES #[(version,date,link,comments),...]

  @staticmethod
  def from_softconf_dump(json_file, conference=None):
    with io.open(json_file, "r", encoding="utf8") as ifh:
      json_str = ifh.read()

    # print (json_str)
    json_data = json.loads(json_str)["submissions"]

    papers = []
    for i in range(len(json_data)):
      reviews = []
      for k in range(len(json_data[i]["reviews"])):
        # print(json_data[i]["reviews"][k])
        review_data = []

        review = Review.from_json_object(json_data[i]["reviews"][k], k==i==0)
        #review = None

        reviews.append(review)

      authors = json_data[i]["authors"] if "authors" in json_data[i] else None
      score = json_data[i]["score"] if "score" in json_data[i] else None
      accepted = json_data[i]["accepted"] if "accepted" in json_data[i] else None
      publication_type = json_data[i]["publication_type"] if "publication_type" in json_data[i] else None
      keywords = json_data[i]["KEYWORDS"] if "KEYWORDS" in json_data[i] else None
      author_emails = json_data[i]["AUTHOR_EMAILS"] if "AUTHOR_EMAILS" in json_data[i] else None
      date_of_submission = json_data[i]["DATE_OF_SUBMISSION"] if "DATE_OF_SUBMISSION" in json_data[i] else None

      paper = Paper(json_data[i]["title"], json_data[i]["abstract"], json_data[i]["id"], reviews, authors, \
              conference, accepted, score, publication_type, None, keywords, author_emails, \
              date_of_submission)

      papers.append(paper)
      # break

    return papers

  @staticmethod
  def from_json(json_filename):
    paper = Paper('', '', None, [])

    datas = []
    with io.open(json_filename, mode='rt', encoding='utf8') as json_file:
      for line in json_file:
      try:
        data = json.loads(line.strip())
        datas.append(data)
      except Exception as e:
        print line
        continue
    if len(datas)==0: return None
    data = datas[-1]

    # Read required fields.
    assert 'title' in data
    assert 'abstract' in data
    paper.TITLE = data['title']
    paper.ABSTRACT = data['abstract']

    if 'id' in data:
      if data['id'] == "":
        paper.ID = json_filename.split("/")[-1].split(".")[0]
      else:
        paper.ID = data['id']
    else:
      paper.ID = json_filename.split("/")[-1].split(".")[0]

    # Read optional fields.
    paper.AUTHORS = data['authors'] if 'authors' in data else None
    paper.CONFERENCE = data['conference'] if 'conference' in data else None
    paper.ACCEPTED = data['accepted'] if 'accepted' in data else None
    paper.SCORE = data['score'] if 'score' in data else None
    paper.PUBLICATION_TYPE = data['publication_type'] if 'publication_type' in data else None
    paper.SCIENCEPARSE = data['scienceparse'] if 'scienceparse' in data else None
    paper.KEYWORDS = data['keywords'] if 'keywords' in data else None
    paper.AUTHOR_EMAILS = data['author_emails'] if 'author_emails' in data else None

    paper.DATE_OF_SUBMISSION = data['DATE_OF_SUBMISSION'] if 'DATE_OF_SUBMISSION' in data else None

    paper.SUBJECTS = data['SUBJECTS'] if 'SUBJECTS' in data else None
    paper.COMMENTS = data['COMMENTS'] if 'COMMENTS' in data else None
    paper.VERSION = data['VERSION'] if 'VERSION' in data else None
    paper.HISTORIES = data['histories'] if 'histories' in data else None

    # Read reviews (mandatory).
    assert 'reviews' in data
    for review_data in data['reviews']:
      review = Review.from_json_object(review_data)
      paper.REVIEWS.append(review)
    return paper



  def to_json_object(self):
    data = dict()

    data["title"] = self.get_title()
    data["abstract"] = self.get_abstract()
    data["id"] = self.get_id()

    if self.AUTHORS is not None:
      data["authors"] = self.get_authors()

    if self.CONFERENCE is not None:
      data["conference"] = self.get_conference()

    if self.ACCEPTED is not None:
      data["accepted"] = self.get_accepted()

    if self.SCORE is not None:
      data["SCORE"] = self.get_score()

    if self.PUBLICATION_TYPE is not None:
      data["publication_type"] = self.get_publication_type()

    if self.SCIENCEPARSE is not None:
      data["SCIENCEPARSE"] = self.get_scienceparse()

    if self.AUTHOR_EMAILS is not None:
      data["AUTHOR_EMAILS"] = self.get_author_emails()

    if self.KEYWORDS is not None:
      data["KEYWORDS"] = self.get_keywords()

    if self.DATE_OF_SUBMISSION is not None:
      data["DATE_OF_SUBMISSION"] = self.get_date_of_submission()

    data["reviews"] = []

    for r in self.get_reviews():
      data["reviews"].append(r.to_json_object())

    # added for arxiv papers

    if self.SUBJECTS is not None:
      data["SUBJECTS"] = self.get_subjects()

    if self.COMMENTS is not None:
      data["COMMENTS"] = self.get_comments()

    if self.VERSION is not None:
      data["VERSION"] = self.get_version()

    data["histories"] = []
    if self.HISTORIES is not None:
      for h in self.get_histories():
        if h is not None:
        v,d,l,p = h
        data["histories"].append((v,d,l, p if p else None))

    return data

  def to_json(self, json_file, mode='a'):

    data = self.to_json_object()

    with open(json_file, mode) as ofh:
      json.dump(data, ofh)
      ofh.write("\n")


  def get_subjects(self):
    return self.SUBJECTS
  def get_comments(self):
    return self.COMMENTS
  def get_version(self):
    return self.VERSION
  def get_histories(self):
    return self.HISTORIES


  def get_title(self):
    return self.TITLE

  def get_abstract(self):
    return self.ABSTRACT

  def abstract_contains_a_term(self, term):
    return (term in self.ABSTRACT)

  def get_id(self):
    return self.ID

  def get_authors(self):
    return self.AUTHORS

  def get_reviews(self):
    return self.REVIEWS

  def get_scienceparse(self):
    return self.SCIENCEPARSE

  def get_title_len(self):
    return len(self.TITLE)

  def get_abstract_len(self):
    return len(self.ABSTRACT)

  def get_conference(self):
    return self.CONFERENCE

  def get_score(self):
    return self.SCORE

  def get_accepted(self):
    return self.ACCEPTED

  def get_publication_type(self):
    return self.PUBLICATION_TYPE

  def get_author_emails(self):
    return self.AUTHOR_EMAILS

  def get_keywords(self):
    return self.KEYWORDS

  def get_date_of_submission(self):
    return self.DATE_OF_SUBMISSION

def main(args):
  papers = Paper.from_softconf_dump('../../data/conll16/reviews.json')
  for paper in papers:
    paper.to_json('../../data/conll16_new/{}.json'.format(paper.ID))

if __name__ == "__main__":
  sys.exit(main(sys.argv))
