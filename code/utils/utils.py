# Close n files to write the processed data into.
def close_n_files(files):
  for file_name in files:
    files[file_name].close()
