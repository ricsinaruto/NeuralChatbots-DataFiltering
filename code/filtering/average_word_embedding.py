import os
import sys
import numpy as np
import requests
import zipfile
from collections import Counter
from clint.textui import progress

from filtering import semantic_clustering


class AverageWordEmbedding(semantic_clustering.SemanticClustering):
  '''
  Averaged word embeddings clustering method. The meaning vector of the
  sentence is created by the weighted average of the word vectors.
  '''

  # Download data from fasttext.
  def download_fasttext(self):
    # Open the url and download the data with progress bars.
    data_stream = requests.get('https://dl.fbaipublicfiles.com/fasttext/' +
      'vectors-english/wiki-news-300d-1M.vec.zip', stream=True)
    zipped_path = os.path.join(self.input_dir, 'fasttext.zip')

    with open(zipped_path, 'wb') as file:
      total_length = int(data_stream.headers.get('content-length'))
      for chunk in progress.bar(data_stream.iter_content(chunk_size=1024),
                                expected_size=total_length / 1024 + 1):
        if chunk:
          file.write(chunk)
          file.flush()

    # Extract file.
    zip_file = zipfile.ZipFile(zipped_path, 'r')
    zip_file.extractall(self.input_dir)
    zip_file.close()

  # Generate a vocab from data files.
  def get_vocab(self, vocab_path):
    vocab = []

    with open(vocab_path, 'w') as file:
      for dp in self.data_points['Source']:
        vocab.extend(dp.string.split())
      file.write('\n'.join(
        [w[0] for w in Counter(vocab).most_common(self.config.vocab_size)]))

  # Download FastText word embeddings.
  def get_fast_text_embeddings(self):
    vocab_path = os.path.join(self.input_dir, 'vocab.txt')
    if not os.path.exists(vocab_path):
      print('No vocab file named \'vocab.txt\' found in ' + self.input_dir)
      print('Building vocab from data.')
      self.get_vocab(vocab_path)

    fasttext_path = os.path.join(self.input_dir, 'wiki-news-300d-1M.vec')
    if not os.path.exists(fasttext_path):
      self.download_fasttext()

    vocab = [line.strip('\n') for line in open(vocab_path)]
    vocab_path = os.path.join(self.input_dir, 'vocab.npy')

    # Save the vectors for words in the vocab.
    with open(fasttext_path, errors='ignore') as in_file:
      with open(vocab_path, 'w') as out_file:
        vectors = {}
        for line in in_file:
          tokens = line.strip().split()
          vectors[tokens[0]] = line

        for word in vocab:
          try:
            out_file.write(vectors[word])
          except KeyError:
            pass

  # Generate the sentence embeddings.
  def generate_embeddings(self, tag, vector_path):
    '''
    Params:
      :tag: Whether it's source or target data.
      :vector_path: Path to save the sentence vectors.
    '''
    vocab = {}
    vocab_path = os.path.join(self.input_dir, 'vocab.npy')
    if not os.path.exists(vocab_path):
      print('File containing word vectors not found in ' + self.input_dir)
      print('The file should be named \'vocab.npy\'')
      print('If you would like to use FastText embeddings press \'y\'')
      if input() == 'y':
        self.get_fast_text_embeddings()
      else:
        sys.exit()

    # Get the word embeddings.
    with open(vocab_path) as v:
      for line in v:
        tokens = line.strip().split()
        vocab[tokens[0]] = [0, np.array(list(map(float, tokens[1:])))]

    embedding_dim = vocab[list(vocab)[0]][1].shape[0]
    unique_sentences = set()
    word_count = 0

    # Statistics of number of words.
    for dp in self.data_points[tag]:
      unique_sentences.add(dp.string)
      for word in dp.string.split():
        if vocab.get(word):
          vocab[word][0] += 1
          word_count += 1

    meaning_vectors = []
    sentences = unique_sentences if self.unique else [
      s.string for s in self.data_points[tag]]
    # Calculate smooth average embedding.
    for s in sentences:
      vectors = []
      for word in s.split():
        vector = vocab.get(word)
        if vector:
          vectors.append(vector[1] * 0.001 / (0.001 + vector[0] / word_count))

      num_vecs = len(vectors)
      if num_vecs:
        meaning_vectors.append(np.sum(np.array(vectors), axis=0) / num_vecs)
      else:
        meaning_vectors.append(np.zeros(embedding_dim))

    np.save(vector_path, np.array(meaning_vectors).reshape(-1, embedding_dim))
