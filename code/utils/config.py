import pickle
import os


# These can also be set as arguments via the command line.
class Config:
  data_dir = 'data/DailyDialog/baseline'  # Directory containing dataset.
  output_dir = 'data/DailyDialog/baseline/filtered_data'
  load_config = None
  source_clusters = 0
  target_clusters = 0
  filter_split = 'full'  # Which data split to filter.
  cluster_type = 'identity'
  unique = False  # Whether to cluster only unique sentences.
  filter_type = 'both'
  min_cluster_size = 4  # Clusters with fewer elements won't get filtered.
  threshold = 1.1  # Entropy threshold for filtering.
  clustering_method = 'mean_shift'  # Kmeans or mean_shift.
  bandwidth = 0.7  # Mean shift bandwidth.
  use_faiss = False  # Whether to use the library for GPU based clustering.
  max_avg_length = 15  # Clusters with longer sentences won't get filtered.
  max_medoid_length = 50  # Clusters with longer medoids won't get filtered.

  # Save this object to output dir.
  def save(self):
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    file = open(os.path.join(self.output_dir, 'config'), 'wb')
    file.write(pickle.dumps(self.__dict__))
    file.close()

  # Load from output dir.
  def load(self):
    file = open(self.load_config, 'rb')
    self.__dict__ = pickle.loads(file.read())
    file.close()
