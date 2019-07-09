from filtering.filter_problem import FilterProblem


class Identity(FilterProblem):
  '''
  Calculate entropy based on and filter individual utterances.
  '''

  # Do the clustering of sources and targets.
  def clustering(self, tag):
    '''
    Params:
      :tag: Whether it's source or target data.
    '''
    rev_tag = 'Target' if tag == 'Source' else 'Source'

    clean_sents = [' '.join(dp.string.split()) for dp in self.data_points[tag]]
    sentence_set = list(set(clean_sents))

    # Build a hash for efficient string searching.
    sentence_dict = {}
    for data_point, clean_sentence in zip(self.data_points[tag], clean_sents):
      if clean_sentence in sentence_dict:
        sentence_dict[clean_sentence].append(data_point)
      else:
        sentence_dict[clean_sentence] = [data_point]

    print(tag + ': ' + str(len(sentence_set)) + ' clusters')

    # Loop through the clusters.
    for i, sentence in enumerate(sentence_set):
      cl = self.ClusterClass(self.DataPointClass(sentence, 10))
      self.clusters[tag].append(cl)

      # Loop through the data points associated with this sentence.
      for data_point in sentence_dict[sentence]:
        data_point.cluster_index = i
        cl.add_element(data_point)
        cl.add_target(self.data_points[rev_tag][data_point.index])
