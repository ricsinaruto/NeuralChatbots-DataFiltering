"""
In this file you can set all tensor2tensor flags, hparams and other settings
for the current run. This file will also be copied to the provided directory.
"""


FLAGS = {
    "data_dir": "data_dir/DailyDialog/no_stop_words",
    "problem": "daily_dialog_chatbot",
    "data_dir": "data_dir/DailyDialog/no_stop_words/filtered_data/avg_word_embedding",
    "filter_problem": "avg_embedding",  # Choose several metrics for clustering.
    "filter_type": "both",  # Can be: target_based, source_based, both.
    "source_clusters": 100,
    "target_clusters": 100,
    "max_length": 0,  # Max length sentences when constructing bigram matrix.
    "min_cluster_size": 2,  # Clusters with fewer elements won't get filtered.
    "num_permutations": 128,  # Only for hash based clustering.
    "character_level": False,  # Only for hash based clustering.
    "treshold": 3,  # Entropy threshold for filtering.
    "ckpt_number": 22001,  # Only for sentence embedding clustering.
    "semantic_clustering_method": "mean_shift",  # Kmeans or mean_shift.
    "mean_shift_bw": 0.7,  # Mean shift bandwidth.
    "use_faiss": False,  # Whether to use the library for GPU based clustering.
    "max_avg_length": 15,  # Clusters with longer sentences won't get filtered.
    "max_medoid_length": 50  # Clusters with longer medoids won't get filtered.

}

PROBLEM_HPARAMS = {
    "num_train_shards": 1,
    "num_dev_shards": 1,
    "vocabulary_size": 16384,
    "dataset_size": 0,  # If zero, take the full dataset.
    "dataset_split": {"train": 80, "val": 10, "test": 10},
    "dataset_version": 2012,  # Only for opensubtitles.
    "name_vocab_size": 3000   # Only for cornell names problem.
}