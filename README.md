# NeuralChatbots-DataFiltering
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Paper](https://img.shields.io/badge/Presented%20at-ACL%202019-yellow.svg)](https://arxiv.org/abs/1905.05471) [![Code1](https://img.shields.io/badge/code-model%20training-green.svg)](https://github.com/ricsinaruto/Seq2seqChatbots) [![Code2](https://img.shields.io/badge/code-evaluation-green.svg)](https://github.com/ricsinaruto/dialog-eval) [![documentation](https://img.shields.io/badge/documentation-on%20wiki-red.svg)](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/wiki) [![twitter](https://img.shields.io/twitter/url/https/shields.io.svg?style=social)](https://ctt.ac/E_jP6)  
A lightweight repo for filtering dialog data with entropy-based methods.

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

## Usage
The main file can be called from anywhere, but when specifying paths to directories you should give them from the root of the repository.
```
python code/main.py -h
```
<a><img src="https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/docs/help.png" align="top" height="800" ></a>

For the complete documentation visit the [wiki](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/wiki).
 
### Visualization
Visualize clustering and filtering results by running the [visualization](https://github.com/ricsinaruto/NeuralChatbots-DataFiltering/blob/master/code/utils/visualization.ipynb) jupyter notebook. The notebook is pretty self-explanatory, you just have to provide the directory containing the clustering files.


## Examples
### See [this](https://anonfile.com/54YeAbf6b6/tables.pdf) for more sample response from [this](https://www.researchgate.net/publication/327594109_Making_Chatbots_Better_by_Training_on_Less_Data) paper.

### Sample responses from various trainings

## Contributing
New clustering methods can also be added, by subclassing the [FilterProblem](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/filter_problem.py) class. This class contains a lot of functionality for clustering and filtering. If your clustering method is similar to others, you will only have to override the clustering function, which does the clustering of the data. Loading and saving data is taken care of, and the clustering should run on the *clusters* and *data_points* lists, which store the data in special [Cluster](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/filter_problem.py) and [DataPoint](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/data_filtering/filter_problem.py) objects. The latter represents one utterance from the dataset, but these can also be subclassed if additional functionality is needed. Finally the new class has to be added to the dictionary in the *data_filtering* function in [run](https://github.com/ricsinaruto/Seq2seqChatbots/blob/master/t2t_csaky/utils/run.py).

## Authors

## License

## Acknowledgments


##### If you require any help with running the code or if you want the files of the trained models, write to this e-mail address. (ricsinaruto@hotmail.com)
