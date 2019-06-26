import argparse

from utils.config import Config
#from filtering.average_word_embedding import AverageWordEmbedding
from filtering.identity import Identity
#from filtering.sent2vec import Sent2vec


def main():
  config = Config()
  parser = argparse.ArgumentParser(
    description='Code for filtering methods in: arxiv.org/abs/1905.05471. ' +
                'These arguments can also be set in config.py, ' +
                'and will be saved to the output directory.')
  parser.add_argument('-d', '--data_dir', default=config.data_dir,
                      help='Directory containing the dataset in these files:' +
                      ' (trainSource.txt, trainTarget.txt, devSource.txt, ' +
                      'devTarget.txt, testSource.txt, testTarget.txt, ' +
                      'vocab.txt)',
                      metavar='')
  parser.add_argument('-o', '--output_dir', default=config.output_dir,
                      help='Save here the filtered data and any output',
                      metavar='')
  parser.add_argument('-l', '--load_config', default=config.load_config,
                      help='Path to load config from file, or leave empty ' +
                      '(default: %(default)s)',
                      metavar='')
  parser.add_argument('-fs', '--filter_split', default=config.filter_split,
                      help='Data split to filter, \'full\' filters ' +
                      'all splits (choices: %(choices)s)',
                      metavar='', choices=['full', 'train', 'dev', 'test'])
  parser.add_argument('-ct', '--cluster_type', default=config.cluster_type,
                      help='Clustering method (choices: %(choices)s)',
                      metavar='',
                      choices=['identity', 'avg_embedding', "sent2vec"])
  parser.add_argument('-sc', '--source_clusters',
                      default=config.source_clusters,
                      help='Number of source clusters in case of Kmeans',
                      metavar='', type=int)
  parser.add_argument('-tc', '--target_clusters',
                      default=config.target_clusters,
                      help='Number of target clusters in case of Kmeans',
                      metavar='', type=int)
  parser.add_argument('-u', '--unique', default=config.unique,
                      help='Whether to cluster only unique sentences ' +
                      '(default: %(default)s)',
                      metavar='', type=bool)
  parser.add_argument('-ft', '--filter_type', default=config.filter_type,
                      help='Filtering way (choices: %(choices)s)',
                      metavar='', choices=['source', 'target', 'both'])
  parser.add_argument('-mins', '--min_cluster_size',
                      default=config.min_cluster_size,
                      help='Clusters with fewer elements won\'t get filtered' +
                      ' (default: %(default)s)',
                      metavar='', type=int)
  parser.add_argument('-t', '--threshold', default=config.threshold,
                      help='Entropy threshold (default: %(default)s)',
                      metavar='', type=int)
  parser.add_argument('-cm', '--clustering_method',
                      default=config.clustering_method,
                      help='Mean shift recommended (choices: %(choices)s)',
                      metavar='', choices=['kmeans, mean_shift'])
  parser.add_argument('-bw', '--bandwidth', default=config.bandwidth,
                      help='Mean shift bandwidth (default: %(default)s)',
                      metavar='', type=float)
  parser.add_argument('-f', '--use_faiss', default=config.use_faiss,
                      help='Whether to use faiss for GPU based clustering ' +
                      '(default: %(default)s)',
                      metavar='', type=bool)
  parser.add_argument('-maxal', '--max_avg_length',
                      default=config.max_avg_length,
                      help='Clusters with higher average sentence length' +
                      'won\'t get filtered (default: %(default)s)',
                      metavar='', type=int)
  parser.add_argument('-maxml', '--max_medoid_length',
                      default=config.max_medoid_length,
                      help='Clusters with longer medoids won\'t get filtered' +
                      ' (default: %(default)s)',
                      metavar='', type=int)

  parser.parse_args(namespace=config)
  if config.load_config:
    config.load()
  config.save()

  filter_problems = {
      'identity': Identity,
      #'avg_embedding': AverageWordEmbedding,
      #'sent2vec': Sent2vec,
  }

  problem = filter_problems[config.cluster_type](config)
  problem.run()


if __name__ == "__main__":
  main()
