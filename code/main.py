import argparse

from config import FLAGS
from utils.utils import save_config_file
from data_filtering.hash_jaccard import HashJaccard
from data_filtering.encoder_state import EncoderState
from data_filtering.average_word_embedding import AverageWordEmbedding
from data_filtering.identity_clustering import IdentityClustering
from data_filtering.unique_clustering import UniqueClustering
from data_filtering.sent2vec import Sent2vec


def main():
  # Create an argument parser.
  parser = argparse.ArgumentParser()
  parser.add_argument('--help', type=str, help='')
  args = parser.parse_args()

  save_config_file(FLAGS["data_dir"])

  filter_problems = {
      "hash_jaccard": HashJaccard,
      "rnn_state": EncoderState,
      "identity": IdentityClustering,
      "avg_embedding": AverageWordEmbedding,
      "sent2vec": Sent2vec,
      "unique_avg_embedding": UniqueClustering,
  }

  problem = filter_problems[FLAGS["filter_problem"]]("full")
  problem.run()


if __name__ == "__main__":
  main()
