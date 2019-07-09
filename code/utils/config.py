import pickle
import os


# These can also be set as arguments via the command line.
class Config:
  data_dir = 'data/DailyDialog/baseline'  # Directory containing dataset.
  output_dir = 'data/DailyDialog/baseline/filtered_data'
  load_config = None
  source_clusters = 50
  target_clusters = 15
  filter_split = 'full'  # Which data split to filter.
  cluster_type = 'avg_embedding'
  unique = False  # Whether to cluster only unique sentences.
  vocab_size = 16384  # Only used for average word embeddings.
  filter_type = 'both'
  min_cluster_size = 4  # Clusters with fewer elements won't get filtered.
  threshold = 1.1  # Entropy threshold for filtering.
  clustering_method = 'kmeans'  # Kmeans or mean_shift.
  bandwidth = 0.7  # Mean shift bandwidth.
  use_faiss = True  # Whether to use the library for GPU based clustering.
  max_avg_length = 15  # Clusters with longer sentences won't get filtered.
  max_medoid_length = 50  # Clusters with longer medoids won't get filtered.
  project_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..')

  # Save this object to output dir.
  def save(self):
    out_dir = os.path.join(self.project_path, self.output_dir)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    file = open(os.path.join(out_dir, 'config'), 'wb')
    file.write(pickle.dumps(self.__dict__))
    file.close()

  # Load from output dir.
  def load(self):
    load_config = os.path.join(self.project_path, self.load_config, 'config')
    file = open(load_config, 'rb')
    self.__dict__ = pickle.loads(file.read())
    file.close()
