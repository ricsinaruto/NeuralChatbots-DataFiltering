import os
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.cluster import MeanShift

from filtering import filter_problem
from utils.config import Config

if Config.use_faiss:
  try:
    from faiss import Kmeans
    Config.faiss = True

  except ImportError:
    print('Failed to import faiss, using SKLearn clustering instead.')

if not Config.use_faiss:
  from sklearn.cluster import KMeans


class DataPoint(filter_problem.DataPoint):
  '''
  A simple class that handles a string example.
  '''
  def __init__(self, string, index, meaning_vector=None):
    '''
    Params:
      :string:  String to be stored.
      :index: Number of the line in the file from which this sentence was read.
      :meaning_vector: Numpy embedding vector for the sentence.
    '''
    super().__init__(string, index)
    self.meaning_vector = meaning_vector


class SemanticClustering(filter_problem.FilterProblem):
  '''
  Base class for the meaning-based (semantic vector representation) clustering.
  The source and target sentences are read into an extended DataPoint object,
  that also contains a 'meaning_vector' attribute. This attribute holds
  the semantic vector representation of the sentence, which will be used
  by the clustering logic.
  '''

  @property
  def DataPointClass(self):
    return DataPoint

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.unique_data = {"Source": [], "Target": []}

  def clustering(self, tag):
    '''
    Params:
      :tag: Whether it's source or target data.
    '''
    data_points = self.unique_data if self.unique else self.data_points
    centroids = self.calculate_centroids(tag)

    data_point_vectors = np.array(
      [data_point.meaning_vector for data_point in
       data_points[tag]]).reshape(
       -1, data_points[tag][0].meaning_vector.shape[-1])

    # Get the actual data point for each centroid.
    tree = BallTree(data_point_vectors)
    _, centroids = tree.query(centroids, k=1)

    # Get the closest centroid for each data point.
    tree = BallTree(data_point_vectors[np.array(centroids).reshape(-1)])
    _, labels = tree.query(data_point_vectors, k=1)
    labels = labels.reshape(-1)

    # Build the list of clusters.
    clusters = {index: self.ClusterClass(data_points[tag][index]) for
                index in {labels[_index] for _index in range(len(labels))}}
    clusters = [(clusters[cluster_index], cluster_index) for cluster_index in
                sorted(list(clusters))]

    label_lookup = {c[1]: i for i, c in enumerate(clusters)}
    clusters = [c[0] for c in clusters]

    rev_tag = 'Target' if tag == 'Source' else 'Source'

    # Store the cluster index for each unique sentence.
    if self.unique:
      cluster_ind_dict = {}
      for data_point, cluster_index in zip(data_points[tag], labels):
        cluster_ind_dict[data_point.string] = label_lookup[cluster_index]

      # This is different for unique clustering.
      for i, data_point in enumerate(self.data_points[tag]):
        cl_index = cluster_ind_dict[data_point.string]
        data_point.cluster_index = cl_index
        clusters[cl_index].add_element(data_point)
        clusters[cl_index].add_target(self.data_points[rev_tag][i])

    # Assign the actual clusters.
    else:
      for dp, cl_index in zip(self.data_points[tag], labels):
        cl_index = label_lookup[cl_index]
        dp.cluster_index = cl_index
        clusters[cl_index].add_element(dp)
        clusters[cl_index].add_target(self.data_points[rev_tag][dp.index])

    self.clusters[tag] = clusters

  def read_inputs(self, tag):
    '''
    Called twice for source and target data. It should implement the
    logic of reading the data from Source and Target files into the
    data_points list. Each sentence should be wrapped into an appropriate
    subclass of the DataPoint class. Source.npy and Target.npy should contain
    sentence embeddings, if not they have to be generated in a subclass.
    These vectors in the .npy files have to correspond to the loaded sentences.

    Params:
      :tag: Whether it's source or target data.
    '''
    super().read_inputs(tag)

    vector_path = os.path.join(self.input_dir, self.tag + tag + '.npy')
    if not os.path.exists(vector_path):
      print('No sentence embeddings found in ' + self.input_dir)
      print('They should be named \'fullSource.npy\' and \'fullTarget.npy\',')
      print('where each line is a vector corresponding to')
      print('sentences in \'fullSource.txt\' and \'fullTarget.txt\'.')
      print('Building sentence representations of ' + self.config.cluster_type)
      self.generate_embeddings(tag, vector_path)

    # Add vectors to sentences.
    sent_vectors = np.load(vector_path)
    if not self.unique:
      for index, dp in enumerate(self.data_points[tag]):
        dp.meaning_vector = sent_vectors[index]
    # Create unique data points if necessary.
    else:
      for i, sent in enumerate(set([s.string for s in self.data_points[tag]])):
        self.unique_data[tag].append(self.DataPointClass(sent,
                                                         i,
                                                         sent_vectors[i]))

  # Has to be implemented by subclass to generate sentence embeddings.
  def generate_embeddings(self, tag):
    raise NotImplementedError

  # Cluster the data and return the centers.
  def calculate_centroids(self, tag):
    '''
    Params:
      :tag: Whether it's source or target data.
    '''
    data_points = self.unique_data if self.unique else self.data_points
    matrix = np.stack([dp.meaning_vector for dp in data_points[tag]])

    if self.config.clustering_method == 'kmeans':
      # Kmeans with either the faiss or the sklearn implementation.
      if self.config.use_faiss:
        kmeans = Kmeans(matrix.shape[1], self.num_clusters[tag], 20, True)
        kmeans.train(matrix)
        centroids = kmeans.centroids
      else:
        kmeans = KMeans(n_clusters=self.num_clusters[tag],
                        random_state=0,
                        n_jobs=10).fit(matrix)
        centroids = kmeans.cluster_centers_

    else:
      mean_shift = MeanShift(bandwidth=self.config.bandwidth, n_jobs=10)
      mean_shift.fit(matrix)
      centroids = mean_shift.cluster_centers_

    return centroids
