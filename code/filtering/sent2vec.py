import sys

from filtering import semantic_clustering


class Sent2vec(semantic_clustering.SemanticClustering):
  def generate_embeddings(self, tag, vector_path):
    print('Currently sent2vec only works with provided sentence embeddings.')
    print('Check github.com/epfml/sent2vec for getting sentence embeddings.')
    print('Btw any kind of sentence embeddings can be used if they are in the')
    print('required format, I recommend github.com/hanxiao/bert-as-service.')
    sys.exit()
