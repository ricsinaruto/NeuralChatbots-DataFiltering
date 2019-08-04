# NeuralChatbots-DataFiltering &middot; [![twitter](https://img.shields.io/twitter/url/https/shields.io.svg?style=social)](https://ctt.ac/E_jP6)
[![Paper](https://img.shields.io/badge/Presented%20at-ACL%202019-yellow.svg)](https://www.aclweb.org/anthology/P19-1567) [![Poster](https://img.shields.io/badge/The-Poster-yellow.svg)](https://ricsinaruto.github.io/website/docs/acl_poster_h.pdf) [![Code1](https://img.shields.io/badge/code-chatbot%20training-green.svg)](https://github.com/ricsinaruto/Seq2seqChatbots) [![Code2](https://img.shields.io/badge/code-evaluation-green.svg)](https://github.com/ricsinaruto/dialog-eval) [![documentation](https://img.shields.io/badge/documentation-on%20wiki-red.svg)](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/wiki) [![blog](https://img.shields.io/badge/Blog-post-black.svg)](https://medium.com/@richardcsaky/neural-chatbots-are-dumb-65b6b40e9bd4)   
A lightweight repo for filtering dialog data with entropy-based methods.  
  
The program **reads the dataset**, runs **clustering** if needed, computes the **entropy** of individual utterances, and then **removes high entropy** utterances based on the threshold, and **saves the filtered dataset** to the output directory. See the [paper](https://www.aclweb.org/anthology/P19-1567) or the [poster](https://ricsinaruto.github.io/website/docs/acl_poster_h.pdf) for more details.

## Features
  :floppy_disk: &nbsp; Cluster and filter any dialog data that you provide, or use pre-downloaded datasets  
  :rocket: &nbsp; Various parameters can be used to adjust the algorithm  
  :ok_hand: &nbsp;&nbsp; Choose between different entropy computation methods  
  :twisted_rightwards_arrows: &nbsp; Choose between different clustering and filtering types  
  :movie_camera: &nbsp; Visualize clustering and filtering results  



## Setup
Run setup.py which installs required packages and steps you through downloading additional data:
```
python setup.py
```
You can download all trained models used in [this](https://www.aclweb.org/anthology/P19-1567) paper from [here](https://mega.nz/#!mI0iDCTI!qhKoBiQRY3rLg3K6nxAmd4ZMNEX4utFRvSby_0q2dwU). Each training contains two checkpoints, one for the validation loss minimum and another after 150 epochs. The data and the trainings folder structure match each other exactly.
## Usage
The main file can be called from anywhere, but when specifying paths to directories you should give them from the root of the repository.
```
python code/main.py -h
```
<a><img src="https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/docs/help.png" align="top" height="800" ></a>    
For the complete **documentation** visit the [wiki](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/wiki).

### Cluster Type
* [identity](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/filtering/identity.py): In this method there is basically no clustering, the entropy of utterances is calculated based on the conditional probability of utterance pairs.
* [avg-embedding](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/filtering/average_word_embedding.py): This clustering type uses average word embedding sentence representations as in [this paper](https://pdfs.semanticscholar.org/3fc9/7768dc0b36449ec377d6a4cad8827908d5b4.pdf).
* [sent2vec](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/filtering/sent2vec.py): This clustering type should use [sent2vec](https://github.com/epfml/sent2vec) sentence embeddings, but currently uses any embeddings you provide to it.

### Filter Type
* **source**: Filters utterance pairs in which the source utterance's entropy is above the threshold.
* **target**: Filters utterance pairs in which the target utterance's entropy is above the threshold.
* **both**: Filters utterance pairs in which either the source or target utterance's entropy is above the threshold.

### [Filtering Demo](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/utils/filtering_demo.ipynb)
In this jupyter notebook you can easily try out the identity filtering method implemented in less than 40 lines, and it filters DailyDialog in a couple of seconds (you only need to provide a sources and targets file). In the second part of the notebook there are some cool visualizations for entropy, frequency and sentence length.  
<a><img src="https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/docs/3d.png" align="top" height="400" ></a>

### [Visualization](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/utils/visualization.ipynb)
Visualize clustering and filtering results by running the [visualization](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/utils/visualization.ipynb) jupyter notebook. The notebook is pretty self-explanatory, you just have to provide the directory containing the clustering files.
<a><img src="https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/docs/visu.png" align="top" height="300" ></a>


## Results & Examples
### High Entropy Utterances and Clusters from [DailyDialog](https://arxiv.org/abs/1710.03957)
<a><img src="https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/docs/cluster_examples.png" align="top" height="500" ></a>
<a><img src="https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/docs/high_entropy.png" align="top" height="500" ></a>  
A high entropy cluster found by sent2vec.

### [Transformer](https://arxiv.org/abs/1706.03762) Trained on [DailyDialog](https://arxiv.org/abs/1710.03957)
For an explanation of the metrics please check [this repo](https://github.com/ricsinaruto/dialog-eval) or the [paper](https://arxiv.org/pdf/1905.05471.pdf).  
<a><img src="https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/docs/metrics_table.png" align="top" height="400" ></a>  
  
<a><img src="https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/docs/example_responses.png" align="top" height="300" ></a>  
More examples can be found in the appendix of the [paper](https://arxiv.org/pdf/1905.05471.pdf).

### [Transformer](https://arxiv.org/abs/1706.03762) Trained on [Cornell](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) and [Twitter](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/twitter)
For an explanation of the metrics please check [this repo](https://github.com/ricsinaruto/dialog-eval) or the [paper](https://arxiv.org/pdf/1905.05471.pdf).  
<a><img src="https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/docs/other_datasets.png" align="top" height="400" ></a> 

## Contributing
##### Check the [issues](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/issues) for some additions where help is appreciated. Any contributions are welcome :heart:
##### Please try to follow the code syntax style used in the repo (flake8, 2 spaces indent, 80 char lines, commenting a lot, etc.)

**New clustering** methods can be added, by subclassing the [FilterProblem](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/filtering/filter_problem.py#L48) class, check [Identity](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/filtering/identity.py) for a minimal example. Normally you only have to redefine the *clustering* function, which does the clustering of sentences.  
  
Loading and saving data is taken care of, and you should use the [Cluster](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/filtering/filter_problem.py#L24) and [DataPoint](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/filtering/filter_problem.py#L9) objects. Use the *data_point* list to get the sentences for your clustering algorithm, and use the *clusters* list to save the results of your clustering. These can also be subclassed if you want to add extra data to your DataPoint and Cluster objects (like a vector).  
  
Finally add your class to the dictionary in [main](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/main.py#L90), and to the command-line argument choices.


## Authors
* **[Richard Csaky](ricsinaruto.github.io)** (If you need any help with running the code: ricsinaruto@hotmail.com)
* **[Patrik Purgai](https://github.com/Mrpatekful)** (clustering part)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/LICENSE) file for details.  
Please include a link to this repo if you use it in your work and consider citing the following paper:
```
@inproceedings{Csaky:2019,
    title = "Improving Neural Conversational Models with Entropy-Based Data Filtering",
    author = "Cs{\'a}ky, Rich{\'a}rd and Purgai, Patrik and Recski, G{\'a}bor",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1567",
    pages = "5650--5669",
}
```
