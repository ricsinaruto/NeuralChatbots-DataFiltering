import sys

from filtering import semantic_clustering


class Sent2vec(semantic_clustering.SemanticClustering):
  def generate_embeddings(self, tag, vector_path):
    print('No sentence embeddings found in ' + self.input_dir)
    print('Currently sent2vec clustering only works if sentence embeddings')
    print('are provided. They should be named \'fullSource.npy\' and')
    print('\'fullTarget.npy\', where each line is a vector corresponding to')
    print('sentences in \'fullSource.txt\' and \'fullTarget.txt\'.')
    print('Check github.com/epfml/sent2vec for getting sentence embeddings.')
    print('Btw any kind of sentence embeddings can be used if they are in the')
    print('required format, I recommend github.com/hanxiao/bert-as-service.')
    sys.exit()
