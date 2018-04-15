
class Review:

  """A review class, contains all bunch of relevant fields"""
  def __init__(self, RECOMMENDATION, COMMENTS, REPLICABILITY=None, PRESENTATION_FORMAT=None, \
         CLARITY=None, MEANINGFUL_COMPARISON=None, SUBSTANCE=None, REVIEWER_CONFIDENCE=None, \
         SOUNDNESS_CORRECTNESS=None, APPROPRIATENESS=None, IMPACT=None, ORIGINALITY=None, OTHER_KEYS=None, \
         IS_META_REVIEW=False, TITLE=None, DATE=None, RECOMMENDATION_UNOFFICIAL=None, IS_ANNOTATED=False):
    self.RECOMMENDATION = RECOMMENDATION
    self.RECOMMENDATION_UNOFFICIAL = RECOMMENDATION_UNOFFICIAL #None # only for aspect prediction
    self.IS_ANNOTATED = IS_ANNOTATED

    self.COMMENTS = COMMENTS
    self.REPLICABILITY = REPLICABILITY
    self.PRESENTATION_FORMAT = PRESENTATION_FORMAT
    self.CLARITY = CLARITY
    self.MEANINGFUL_COMPARISON = MEANINGFUL_COMPARISON
    self.SUBSTANCE = SUBSTANCE
    self.REVIEWER_CONFIDENCE = REVIEWER_CONFIDENCE
    self.SOUNDNESS_CORRECTNESS = SOUNDNESS_CORRECTNESS
    self.APPROPRIATENESS = APPROPRIATENESS
    self.IMPACT = IMPACT
    self.ORIGINALITY = ORIGINALITY
    self.OTHER_KEYS = OTHER_KEYS
    self.IS_META_REVIEW = IS_META_REVIEW
    self.TITLE = TITLE
    self.DATE = DATE

  @staticmethod
  def get_json_string(json_object, string, missing_fields):
    if string in json_object:
      return json_object[string]
    elif missing_fields is not None:
      missing_fields.append(string)

    return None

  @staticmethod
  def from_json_object(json_object, print_missing_fields=False):
    assert "comments" in json_object
    comments = json_object["comments"]

    missing_fields = None

    if print_missing_fields:
      missing_fields = []

    recommendation = Review.get_json_string(json_object, "RECOMMENDATION", missing_fields)


    recommendation_unofficial = Review.get_json_string(json_object, "RECOMMENDATION_UNOFFICIAL", missing_fields)

    is_annotated = Review.get_json_string(json_object, "IS_ANNOTATED", missing_fields)

    replicability = Review.get_json_string(json_object, "REPLICABILITY", missing_fields)
    clarity = Review.get_json_string(json_object, "CLARITY", missing_fields)
    substance = Review.get_json_string(json_object, "SUBSTANCE", missing_fields)
    appropriateness = Review.get_json_string(json_object, "APPROPRIATENESS", missing_fields)
    originality = Review.get_json_string(json_object, "ORIGINALITY", missing_fields)
    presentation_format = Review.get_json_string(json_object, "PRESENTATION_FORMAT", missing_fields)
    meaningful_comparison = Review.get_json_string(json_object, "MEANINGFUL_COMPARISON", missing_fields)
    reviewer_confidence = Review.get_json_string(json_object, "REVIEWER_CONFIDENCE", missing_fields)
    soundness_correctness = Review.get_json_string(json_object, "SOUNDNESS_CORRECTNESS", missing_fields)
    impact = Review.get_json_string(json_object, "IMPACT", missing_fields)
    is_meta_review = Review.get_json_string(json_object, "IS_META_REVIEW", missing_fields)
    date = Review.get_json_string(json_object, "DATE", missing_fields)
    title = Review.get_json_string(json_object, "TITLE", missing_fields)
    other_keys = Review.get_json_string(json_object, "OTHER_KEYS", missing_fields)

    if print_missing_fields and len(missing_fields):
      print("The following fields are missing in json input file:",missing_fields)
    return Review(recommendation, comments, replicability, presentation_format, clarity, meaningful_comparison, \
            substance, reviewer_confidence, soundness_correctness, appropriateness, impact, originality, \
            other_keys, is_meta_review, title, date, recommendation_unofficial, is_annotated )

  def to_json_object(self):
    data = dict()

    data["comments"] = self.get_comments().decode('cp1252', errors='ignore').encode('utf-8')

    if self.RECOMMENDATION is not None:
      data["RECOMMENDATION"] = self.get_recommendation()

    if self.RECOMMENDATION_UNOFFICIAL is not None:
      data["RECOMMENDATION_UNOFFICIAL"] = self.get_recommendation_unofficial()
    if self.IS_ANNOTATED is not None:
      data["IS_ANNOTATED"] = self.get_is_annotated()


    if self.REPLICABILITY is not None:
      data["REPLICABILITY"] = self.get_replicability()
    if self.PRESENTATION_FORMAT is not None:
      data["PRESENTATION_FORMAT"] = self.get_presentation_format()
    if self.CLARITY is not None:
      data["CLARITY"] = self.get_clarity()
    if self.MEANINGFUL_COMPARISON is not None:
      data["MEANINGFUL_COMPARISON"] = self.get_meaningful_comparison()
    if self.SUBSTANCE is not None:
      data["SUBSTANCE"] = self.get_substance()
    if self.REVIEWER_CONFIDENCE is not None:
      data["REVIEWER_CONFIDENCE"] = self.get_reviewer_confidence()
    if self.SOUNDNESS_CORRECTNESS is not None:
      data["SOUNDNESS_CORRECTNESS"] = self.get_soundness_correctness()
    if self.APPROPRIATENESS is not None:
      data["APPROPRIATENESS"] = self.get_appropriateness()
    if self.IMPACT is not None:
      data["IMPACT"] = self.get_impact()
    if self.ORIGINALITY is not None:
      data["ORIGINALITY"] = self.get_originality()
    if self.OTHER_KEYS is not None:
      data["OTHER_KEYS"] = self.get_other_keys()
    if self.IS_META_REVIEW is not None:
      data["IS_META_REVIEW"] = self.is_meta_review()
    if self.TITLE is not None:
      data["TITLE"] = self.get_title()
    if self.DATE is not None:
      data["DATE"] = self.get_date()


    return data

  def get_recommendation(self):
    return self.RECOMMENDATION

  def get_recommendation_unofficial(self):
    return self.RECOMMENDATION_UNOFFICIAL

  def get_is_annotated(self):
    return self.IS_ANNOTATED

  def get_comments(self):
    return self.COMMENTS

  def get_replicability(self):
    return self.REPLICABILITY

  def get_presentation_format(self):
    return self.PRESENTATION_FORMAT

  def get_clarity(self):
    return self.CLARITY

  def get_meaningful_comparison(self):
    return self.MEANINGFUL_COMPARISON

  def get_substance(self):
    return self.SUBSTANCE

  def get_reviewer_confidence(self):
    return self.REVIEWER_CONFIDENCE

  def get_soundness_correctness(self):
    return self.SOUNDNESS_CORRECTNESS

  def get_appropriateness(self):
    return self.APPROPRIATENESS

  def get_impact(self):
    return self.IMPACT

  def get_originality(self):
    return self.ORIGINALITY

  def get_other_keys(self):
    return self.OTHER_KEYS

  def is_meta_review(self):
    return self.IS_META_REVIEW

  def get_title(self):
    return self.TITLE

  def get_date(self):
    return self.DATE
