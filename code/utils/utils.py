# Temporary helper function to load a vocabulary.
def load_vocab(config):
  vocab = open()
  vocab_dict = {}
  # Read the vocab file.
  i = 0
  for word in vocab:
    vocab_dict[word.strip("\n")] = i
    i += 1

  vocab.close()
  return vocab_dict


# Close n files to write the processed data into.
def close_n_files(files):
  for file_name in files:
    files[file_name].close()
