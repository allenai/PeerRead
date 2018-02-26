import spacy
import re

class ScienceParse:
  """A review class, contains all bunch of relevant fields"""
  def __init__(self, title, abstract, sections, reference_titles, reference_venues, reference_years, reference_mention_contexts,
         reference_num_mentions, authors=None, emails = None, other_keys=None):
    self.title = title
    self.abstract = abstract
    self.sections = sections
    self.reference_titles = reference_titles
    self.reference_venues = reference_venues
    self.reference_years = reference_years
    self.reference_mention_contexts = reference_mention_contexts
    self.reference_num_mentions = reference_num_mentions
    self.authors = authors
    self.emails = emails

  def get_sections_dict(self):
    return self.sections

  def get_reference_title_dict(self):
    return self.reference_titles

  def get_reference_venues_dict(self):
    return self.reference_venues

  def get_reference_years_dict(self):
    return self.reference_years

  def get_reference_mention_contexts_dict(self):
    return self.reference_mention_contexts

  def get_reference_num_mentions_dict(self):
    return self.reference_num_mentions

  def get_num_references(self):
    return len(self.get_reference_years_dict())

  def get_num_refmentions(self):
    num_refmentions = 0
    for refid in self.reference_num_mentions:
      num_refmentions  = num_refmentions + self.reference_num_mentions[refid]
    return num_refmentions

  def get_most_recent_reference_year(self):
    most_recent = 0
    for refid in self.reference_years:
      if self.reference_years[refid] > most_recent:
        most_recent = self.reference_years[refid]
    return most_recent

  def get_avg_length_reference_mention_contexts(self):
    sum_length = 0.0
    for refid in self.reference_mention_contexts:
      sum_length = sum_length + len(self.reference_mention_contexts[refid])
    avg_length = 0
    if len(self.reference_mention_contexts) > 0:
      avg_length = sum_length / len(self.reference_mention_contexts)
    return avg_length

  def get_paper_content(self):
    content = self.title + " " + self.abstract + " " + self.get_author_names_string() + " " + \
          self.get_domains_from_emails()
    for sect_id in sorted(self.sections):
      # print("###",str(sect_id))
      content = content + " " +  self.sections[sect_id]
    content = re.sub("\n([0-9]*\n)+", "\n", content)
    return content

  def get_tagged_paper_content(self):
    content = self.get_paper_content()

    nlp = spacy.load('en', parser=False)

    doc = nlp(content)

    return " ".join([x.text+"_"+x.tag_ for x in doc])

  def get_frequent_words_proportion(self, hfws, most_frequent_words, least_frequent_words):
    content = self.get_paper_content().split()

    n = 0
    t = 0
    # print(str(most_frequent_words).encode('utf8'))
    for w in content:
      if w not in hfws and w not in least_frequent_words:
        t += 1
        n += w in most_frequent_words

    # print (n,len(content),1.*n/t)

    return 1.*n/t

  # #papers referred from -5 years from year of submission
  def get_num_recent_references(self, submission_year):
    num_recent_references = 0
    for refid in self.reference_years:
      if (submission_year - self.reference_years[refid] < 5):
        num_recent_references = num_recent_references + 1
    return num_recent_references

  # word offset of figure 1
  def get_word_offset_of_first_fig_reference(self):
    content_words = self.get_paper_content().split(" ")
    indices = [i for i, x in enumerate(content_words) if x == "Figure"]
    return indices[0]

  # num references to #figures
  def get_num_ref_to_figures(self):
    content_words = self.get_paper_content().split(" ")
    figure_indices = [i for i, x in enumerate(content_words) if x == "Figure"]
    return len(figure_indices)

  # num references to #tables
  def get_num_ref_to_tables(self):
    content_words = self.get_paper_content().split(" ")
    table_indices = [i for i, x in enumerate(content_words) if x == "Table"]
    return len(table_indices)

  # # of references to Section
  def get_num_ref_to_sections(self):
    content_words = self.get_paper_content().split(" ")
    section_indices = [i for i, x in enumerate(content_words) if x == "Section"]
    return len(section_indices)

  # related work at front/back
  # #unique words
  def get_num_uniq_words(self):
    return len(set(self.get_paper_content().split(" ")))

  # num of sections
  def get_num_sections(self):
    return len(self.sections)

  # avg length of sentences
  def get_avg_sentence_length(self):
    sentences = self.get_paper_content().split(". ")
    sentence_lengths = [len(s.split(" ")) for s in sentences]
    return (1.0 * sum(sentence_lengths))/len(sentence_lengths)

  # whether paper has appendix
  def get_contains_appendix(self):
    content_words = self.get_paper_content().split(" ")
    figure_indices = [i for i, x in enumerate(content_words) if x == "Appendix"]
    return int(len(figure_indices) > 0)

  # publishing a dataset / code
  def get_contains_appendix(self):
    content_words = self.get_paper_content().split(" ")
    figure_indices = [i for i, x in enumerate(content_words) if x == "Appendix"]
    return int(len(figure_indices) > 0)

  # #authors
  def get_num_authors(self):
    if self.authors == None:
      return 0
    return len(self.authors)

  # get author names as a string
  def get_author_names_string(self):
    if self.authors == None:
      return ""
    return str.join(' ', self.authors)

  # get domains from emails
  def get_domains_from_emails(self):
    domains = []
    for email in self.emails:
      domains.append(email.split('@')[1].replace(".", "_"))
    return str.join(' ', domains)

  # num references to equations
  def get_num_ref_to_equations(self):
    content_words = self.get_paper_content().split(" ")
    equation_indices = [i for i, x in enumerate(content_words) if x == "Equation"]
    return len(equation_indices)

  # num references to theorems
  def get_num_ref_to_theorems(self):
    content_words = self.get_paper_content().split(" ")
    theorem_indices = [i for i, x in enumerate(content_words) if x == "Theorem"]
    return len(theorem_indices)
