import pickle
import os


# These can also be set as arguments via the command line.
class Config:
  data_dir = "data/DailyDialog/baseline"  # Directory containing dataset.
  output_dir = "data/DailyDialog/baseline/filtered_data"
  dataset_split = {"train": 80, "val": 10, "test": 10}
  cluster_type = "avg_embedding"
  unique = False  # Whether to cluster only unique sentences.
  filter_type = "both"
  min_cluster_size = 2  # Clusters with fewer elements won't get filtered.
  treshold = 3  # Entropy threshold for filtering.
  clustering_method = "mean_shift"  # Kmeans or mean_shift.
  bandwidth = 0.7  # Mean shift bandwidth.
  use_faiss = False  # Whether to use the library for GPU based clustering.
  max_avg_length = 15  # Clusters with longer sentences won't get filtered.
  max_medoid_length = 50  # Clusters with longer medoids won't get filtered.

  def save(self):
    file = open(os.path.join(self.output_dir, 'config'), 'wb')
    file.write(pickle.dumps(self.__dict__))
    file.close()

  def load(self):
    file = open(os.path.join(self.output_dir, 'config'), 'rb')
    self.__dict__ = pickle.loads(file.read())
    file.close()
