import os
import math
from collections import Counter

from utils import utils


class DataPoint:
  '''
  A simple class that handles a string example.
  '''
  def __init__(self, string, index):
    '''
    Params:
      :string: String to be stored.
      :index: Number of the line in the file from which this sentence was read.
    '''
    self.index = index
    self.string = string.strip('\n')
    self.cluster_index = 0


class Cluster:
  '''
  A class to handle one cluster in the clustering problem.
  '''
  def __init__(self, medoid):
    '''
    Params:
      :medoid: Center of the cluster: a DataPoint object.
    '''
    self.medoid = medoid
    self.elements = []
    self.targets = []
    self.entropy = 0
    self.index = 0

  # Append an element to the list of elements in the cluster.
  def add_element(self, element):
    self.elements.append(element)

  # append an element to the list of targets in the cluster.
  def add_target(self, target):
    self.targets.append(target)


class FilterProblem:
  '''
  An abstract class to handle different types of filtering.
  '''

  @property
  def DataPointClass(self):
    return DataPoint

  @property
  def ClusterClass(self):
    return Cluster

  def __init__(self, config):
    '''
    Params:
      :config: Config object storing all arguments.
    '''
    self.config = config
    self.tag = config.filter_split
    self.threshold = config.threshold
    self.max_avg_length = config.max_avg_length
    self.max_medoid_length = config.max_medoid_length
    self.min_cluster_size = config.min_cluster_size

    self.project_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), '..', '..')
    self.output_dir = os.path.join(self.project_path, config.output_dir)
    self.input_dir = os.path.join(self.project_path, config.data_dir)
    self.type = config.filter_type

    self.clusters = {'Source': [], 'Target': []}
    self.data_points = {'Source': [], 'Target': []}
    self.num_clusters = {'Source': config.source_clusters,
                         'Target': config.target_clusters}

    self.build('Source')
    self.build('Target')

  # Build statistics and full files.
  def build(self, tag):
    '''
    Params:
      :tag: 'Source' or 'Target'.
    '''
    splits = ['train', 'dev', 'test']
    files = [open(os.path.join(self.input_dir, split + tag + '.txt')).read()
             for split in splits]

    full_path = os.path.join(self.input_dir, 'full' + tag + '.txt')
    if not os.path.exists(full_path):
      open(full_path, 'w').write(''.join(files))

    if tag == 'Source':
      self.line_counts = dict(zip(splits,
                                  map(lambda x: len(x.split('\n')), files)))

  # Main method that will run all the functions to do the filtering.
  def run(self):
    # If we have already done the clustering, don't redo it.
    source_data = os.path.join(self.output_dir,
                               self.tag + 'Source_cluster_elements.txt')
    target_data = os.path.join(self.output_dir,
                               self.tag + 'Target_cluster_elements.txt')
    if os.path.exists(source_data) and os.path.exists(target_data):
      print('Cluster files are in ' + self.output_dir + ', filtering now.')
      self.load_clusters(source_data, target_data)
      self.filtering()

    else:
      print('No cluster files in ' + self.output_dir + ', clustering now.')
      self.read_inputs('Source')
      self.read_inputs('Target')
      self.clustering('Source')
      self.clustering('Target')
      self.save_clusters('Source')
      self.save_clusters('Target')
      self.filtering()

  # Read the data and make it ready for clustering.
  def read_inputs(self, tag):
    '''
    Params:
      :tag: 'Source' or 'Target'.
    '''
    file = open(os.path.join(self.input_dir, self.tag + tag + '.txt'))
    for i, line in enumerate(file):
      self.data_points[tag].append(self.DataPointClass(line, i))
    file.close()

    print('Finished reading ' + tag + ' data.')

  # Load clusters from file.
  def load_clusters(self, source_path, target_path):
    '''
    Params:
      :source_path: Path to source cluster elements.
      :target_path: Path to target cluster elements.
    '''
    source_clusters = {}
    target_clusters = {}
    source_data_points = {}
    target_data_points = {}

    with open(source_path, 'r') as source_file:
      for line in source_file:
        # If the data contains these special characters it won't work.
        [source_index_center, source_target, _] = line.split('<=====>')
        [source_index, source_center] = source_index_center.split(';')
        [source, target] = source_target.split('=')

        # Initialize the source and target utterances.
        source_data_points[int(source_index)] = self.DataPointClass(
          source, int(source_index))
        target_data_points[int(source_index)] = self.DataPointClass(
          target, int(source_index))

        # If this is a new cluster add it to the list.
        if source_clusters.get(source_center) is None:
          center = self.DataPointClass(source_center, 0)
          source_clusters[source_center] = self.ClusterClass(center)
          source_clusters[source_center].index = len(source_clusters) - 1

        # Add the elements to the cluster.
        source_data_points[int(source_index)].cluster_index = \
          source_clusters[source_center].index
        source_clusters[source_center].add_element(
          source_data_points[int(source_index)])
        source_clusters[source_center].add_target(
          target_data_points[int(source_index)])

    with open(target_path, 'r') as target_file:
      for line in target_file:
        [target_index_center, target_source, _] = line.split('<=====>')
        [target_index, target_center] = target_index_center.split(';')
        [target, source] = target_source.split('=')

        # All elements are already added at this point.
        target_data_point = target_data_points[int(target_index)]
        source_data_point = source_data_points[int(target_index)]

        # If this is a new cluster add it to the list.
        if target_clusters.get(target_center) is None:
          center = self.DataPointClass(target_center, 0)
          target_clusters[target_center] = self.ClusterClass(center)
          target_clusters[target_center].index = len(target_clusters) - 1

        # Add the elements to the cluster.
        target_data_point.cluster_index = target_clusters[target_center].index
        target_clusters[target_center].add_element(target_data_point)
        target_clusters[target_center].add_target(source_data_point)

    # Save the data correctly into self.clusters.
    def id(cl):
      return sorted(list(cl), key=lambda x: cl[x].index)
    self.clusters['Source'] = [source_clusters[i] for i in id(source_clusters)]
    self.clusters['Target'] = [target_clusters[i] for i in id(target_clusters)]

  # Cluster sources or targets, should be implemented in subclass.
  def clustering(self, tag):
    raise NotImplementedError

  # Return a list of indices, showing which clusters should be filtered out.
  def get_filtered_indices(self, tag):
    '''
    Params:
      :tag: Source or Target.
    '''
    indices = []
    for num_cl, cluster in enumerate(self.clusters[tag]):
      # Build a distribution for the current cluster, based on the targets.
      distribution = Counter([t.cluster_index for t in cluster.targets])

      num_elements = len(cluster.elements)
      # Calculate entropy.
      entropy = 0
      for cl_index in distribution:
        if num_elements > 1:
          probability = distribution[cl_index] / num_elements
          entropy += probability * math.log(probability, 2)
      cluster.entropy = -entropy

      avg_length = (
          sum(len(sent.string.split()) for sent in cluster.elements) /
          (num_elements if num_elements > 0 else 1))
      medoid_length = len(cluster.medoid.string.split())

      # Filter based on threshold.
      if (cluster.entropy > self.threshold and
          avg_length < self.max_avg_length and
          medoid_length < self.max_medoid_length):
        indices.append(num_cl)

    print('Finished filtering ' + tag + ' data.')
    return indices

  # Do the filtering of the dataset.
  def filtering(self):
    # These are not needed anymore.
    self.data_points['Source'].clear()
    self.data_points['Target'].clear()

    # Get the filtered indices for both sides.
    source_indices = self.get_filtered_indices('Source')
    target_indices = self.get_filtered_indices('Target')

    file_dict = {}
    # We have to open 6 files in this case.
    if self.tag == 'full':
      name_list = ['trainS', 'trainT', 'devS', 'devT', 'testS', 'testT']
      file_dict = dict(zip(name_list, self.open_6_files()))
    else:
      file_dict[self.tag + 'S'] = open(
        os.path.join(self.output_dir, self.tag + 'Source.txt'), 'w')
      file_dict[self.tag + 'T'] = open(
        os.path.join(self.output_dir, self.tag + 'Target.txt'), 'w')

    # Handle all other cases and open files.
    if self.type == 'source' or self.type == 'both':
      file_dict['source_entropy'] = open(
        os.path.join(self.output_dir,
                     self.tag + 'Source_cluster_entropies.txt'), 'w')
    if self.type == 'target' or self.type == 'both':
      file_dict['target_entropy'] = open(
        os.path.join(self.output_dir,
                     self.tag + 'Target_cluster_entropies.txt'), 'w')

    # Save data and close files.
    self.save_filtered_data(source_indices, target_indices, file_dict)
    utils.close_n_files(file_dict)

  # Save the new filtered datasets.
  def save_filtered_data(self, source_indices, target_indices, file_dict):
    '''
    Params:
      :source_indices: Indices of source clusters that will be filtered.
      :target_indices: Indices of target clusters that will be filtered.
      :file_dict: Dictionary containing all the files that we want to write.
    '''
    # Function for writing filtered source or target data to file.
    def save_dataset(tag):
      for num_cl, cluster in enumerate(self.clusters[tag]):
        # Write cluster entropies.
        file_dict[tag.lower() + '_entropy'].write(
            cluster.medoid.string + ';' +
            str(cluster.entropy) + ';' +
            str(len(cluster.elements)) + '\n')

        # Check if a cluster is smaller than threshold.
        cluster_too_small = len(cluster.elements) < self.min_cluster_size
        indices = source_indices if tag == 'Source' else target_indices

        # Make sure that in 'both' case this is only run once.
        if ((tag == 'Source' or self.type != 'both') and
            (num_cl not in indices or cluster_too_small)):
          # Filter one side.
          for num_el, element in enumerate(cluster.elements):
            target_cl = cluster.targets[num_el].cluster_index
            if self.type == 'both':
              cluster_too_small = (
                len(self.clusters['Target'][target_cl].elements) <
                self.min_cluster_size)
            # Check both sides in 'both' case.
            if ((target_cl not in target_indices or cluster_too_small) or
                self.type != 'both'):

              # Reverse if Target.
              source = element.string + '\n'
              target = cluster.targets[num_el].string + '\n'
              source_string = source if tag == 'Source' else target
              target_string = target if tag == 'Source' else source

              # Separate the full case.
              if self.tag == 'full':
                if element.index < self.line_counts['train']:
                  file_dict['trainS'].write(source_string)
                  file_dict['trainT'].write(target_string)
                elif element.index < (self.line_counts['train'] +
                                      self.line_counts['dev']):
                  file_dict['devS'].write(source_string)
                  file_dict['devT'].write(target_string)
                else:
                  file_dict['testS'].write(source_string)
                  file_dict['testT'].write(target_string)
              else:
                file_dict[self.tag + 'S'].write(source_string)
                file_dict[self.tag + 'T'].write(target_string)

    # Write source entropies and data to file.
    if self.type == 'source' or self.type == 'both':
      save_dataset('Source')
    # Write target entropies and data to file.
    if self.type == 'target' or self.type == 'both':
      save_dataset('Target')

  # Save clusters and their elements to files.
  def save_clusters(self, tag):
    '''
    Params:
      :tag: Whether it's source or target data.
    '''
    output = open(
      os.path.join(self.output_dir,
                   self.tag + tag + '_cluster_elements.txt'), 'w')

    medoid_counts = []
    rev_tag = 'Target' if tag == 'Source' else 'Source'

    for cluster in self.clusters[tag]:
      medoid_counts.append((cluster.medoid.string, len(cluster.elements)))

      # Save together the source and target medoids and elements.
      for source, target in zip(cluster.elements, cluster.targets):
        output.write(
            str(source.index) + ';' +
            cluster.medoid.string + '<=====>' +
            source.string + '=' +
            target.string + '<=====>' +
            self.clusters[rev_tag][target.cluster_index].medoid.string + ':' +
            str(target.cluster_index) + '\n')
    output.close()

    # Save the medoids and the count of their elements, in decreasing order.
    output = open(os.path.join(self.output_dir,
                               self.tag + tag + '_clusters.txt'), 'w')
    medoids = sorted(medoid_counts, key=lambda count: count[1], reverse=True)

    for medoid in medoids:
      output.write(medoid[0] + ':' + str(medoid[1]) + '\n')
    output.close()

    if tag == 'Target':
      print('Finished clustering, proceeding with filtering.')

  # Open the 6 files.
  def open_6_files(self):
    trainS = open(os.path.join(self.output_dir, 'trainSource.txt'), 'w')
    trainT = open(os.path.join(self.output_dir, 'trainTarget.txt'), 'w')
    devS = open(os.path.join(self.output_dir, 'devSource.txt'), 'w')
    devT = open(os.path.join(self.output_dir, 'devTarget.txt'), 'w')
    testS = open(os.path.join(self.output_dir, 'testSource.txt'), 'w')
    testT = open(os.path.join(self.output_dir, 'testTarget.txt'), 'w')

    return [trainS, trainT, devS, devT, testS, testT]
