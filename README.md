# NeuralChatbots-DataFiltering &middot; [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Paper](https://img.shields.io/badge/Presented%20at-ACL%202019-yellow.svg)](https://arxiv.org/abs/1905.05471) [![Code1](https://img.shields.io/badge/code-model%20training-green.svg)](https://github.com/ricsinaruto/Seq2seqChatbots) [![Code2](https://img.shields.io/badge/code-evaluation-green.svg)](https://github.com/ricsinaruto/dialog-eval) [![documentation](https://img.shields.io/badge/documentation-on%20wiki-red.svg)](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/wiki)
A lightweight repo for filtering dialog data with entropy-based methods.

## Features


## Setup
Run setup.py which installs required packages and steps you through downloading additional data:
```
python setup.py
```

## Usage
In order to run something, you will have to call the [main](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/main.py) file:
```
python t2t_csaky/main.py --mode=train
```
The mode flag can be one of the following four: *{[generate_data](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#generate-data), [filter data](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#filter-data), [train](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#train), [decode](https://github.com/ricsinaruto/Seq2seqChatbots/tree/master#decode)}*. Additionally an *experiment* mode can be used, where you can speficy what to do inside the *experiment* function of the *[run](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/utils/run.py)* file. A detailed explanation is given lower, for what each mode does. With version v1.1 the main and config files were introduced, for a more streamlined experience, but if you want more freedom and want to use tensor2tensor commands directly, check the v1.0_README for the old way.
#### [Config](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/config.py)
You can control the flags and parameters of each mode directly in this file. Furthermore, for each run that you initiate this file will be copied to the appropriate directory, so you can quickly access the parameters of any run. There are some flags that you have to set for every mode (the *FLAGS* dictionary in the config file):
* **t2t_usr_dir**: Path to the directory where my code resides. You don't have to change this, unless you rename the directory.
* **data_dir**: The path to the directory where you want to generate the source and target pairs, and other data. The dataset will be downloaded one level higher from this directory into a *raw_data* folder.
* **problem**: This is the name of a registered problem that tensor2tensor needs. Detailed in the *generate_data* section below.
 
### Filter Data
Run this mode if you want to filter a dataset based on entropy as described [here](https://www.researchgate.net/publication/327594109_Making_Chatbots_Better_by_Training_on_Less_Data). You can choose from several working clustering methods:
* *[hash_jaccard](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/hash_jaccard.py)*: Cluster sentences based on the jaccard similarity between them, using the [datasketch](https://github.com/ekzhu/datasketch) library.
* *[identity_clustering](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/identity_clustering.py)*: This is a very simple clustering method, where only sentences that are exactly the same (syntactically) fall into one cluster.
* *[average_word_embedding](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/average_word_embedding.py)*: More sophisticated method where sentences are clustered based on their average word embedding representation.
* *[encoder_state](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/identity_clustering.py)*: Cluster sentences based on their representation from a trained seq2seq model's encoder RNN final hidden state.

The *DATA_FILTERING* dictionary in the config file contains the parameters for this mode, which you will have to set. Short explanation:
* *data_dir*: Specify the directory where the new dataset will be saved.
* *filter_problem*: Specify the name of the clustering method, can be one of the above.
* *filter_type*: Whether to filter source, target, or both sides.
* *treshold*: The entropy treshold above which source-target pairs will get filtered.
* *semantic_clustering_method*: Whether to use Kmeans or Mean shift for the semantic clustering types. Mean shift looks like the superior method, where only a radius has to be given.

Some results of the clustering/filtering methods can be seen in the *[filtering_visualization](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/scripts/filtering_visualization.ipynb)* jupyter notebook.
New clustering methods can also be added, by subclassing the [FilterProblem](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/filter_problem.py) class. This class contains a lot of functionality for clustering and filtering. If your clustering method is similar to others, you will only have to override the clustering function, which does the clustering of the data. Loading and saving data is taken care of, and the clustering should run on the *clusters* and *data_points* lists, which store the data in special [Cluster](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/filter_problem.py) and [DataPoint](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/filter_problem.py) objects. The latter represents one utterance from the dataset, but these can also be subclassed if additional functionality is needed. Finally the new class has to be added to the dictionary in the *data_filtering* function in [run](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/utils/run.py).

## Examples
### See [this](https://anonfile.com/54YeAbf6b6/tables.pdf) for more sample response from [this](https://www.researchgate.net/publication/327594109_Making_Chatbots_Better_by_Training_on_Less_Data) paper.

### Sample responses from various trainings

## Contributing

## Authors

## License

## Acknowledgments


##### If you require any help with running the code or if you want the files of the trained models, write to this e-mail address. (ricsinaruto@hotmail.com)
