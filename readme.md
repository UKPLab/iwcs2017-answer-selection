# Representation Learning for Answer Selection with LSTM-Based Importance Weighting

The source code in this project allows you to replicate our results reported in the 
associated paper. The project can also be used as a starting point for new experiments.

Please use the following citation:

```
@InProceedings{rueckle:2017:IWCS,
  title = {Representation Learning for Answer Selection with LSTM-Based Importance Weighting},
  author = {R{\"u}ckl{\'e}, Andreas and Gurevych, Iryna},
  publisher = {Association for Computational Linguistics},
  booktitle = {Proceedings of the 12th International Conference on Computational Semantics (IWCS 2017)},
  month = sep,
  year = {2017},
  location = {Montpellier, France}
}
```

Contact person: Andreas Rücklé

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 



# Experimental Framework

In the following, we briefly describe the different components that are
included in this project and list the steps required to run LW experiments.

## Project Structure

The project includes the following files and folder:

  - __/configs__: A folder that contains configuration files that can be used
		to run the experiments of the associated paper
  - __/experiment__: Contains the source code of all models, the training
		procedure and the evaluation
  - __/scripts__: Contains additional scripts for hyperparameter optimization
  - __/tests__: Unit tests
  - __default_config.yaml__: The standard configuration fallback
  - __requirements.txt__: Lists the required python packages. For information
  		on the required tensorflow versions, please see the notes below
  - __run_experiment.py__: The main entry point for each experiment


## Configuration Files

Each experiment requires a dedicated YAML configuration file. It defines which
components should be loaded for execution. Each experiment can define four
components:

  1. Data: Defines which data loading module should be used. We included
		modules for InsuranceQA version 1 and 2 as well as WikiQA
  2. Model: The model which should be used for training and evaluation
  3. Training: The training module implements the training procedure
  4. Evaluation (optional): Performs the final evaluation

Each module can be configured in separate sections of the configuration file.
The configurations used in our experiments can be found in the folder __/configs__.


## Running Experiments

We used Python 2.7.6 with TensorFlow 0.11.0rc0 on a Nvidia Tesla K40 GPU for
all InsuranceQA experiments, and Python 3.5.3 with TensorFlow 0.10.0 on a
Nvidia K20x GPU for all WikiQA experiments.

To run the experiments:

  - Install all requirements listed in requirements.txt (_> pip install
		requirements.txt_)
  - Install TensorFlow
  - Download InsuranceQA (https://github.com/shuzi/insuranceQA)
  - Download GloVe Embeddings and unzip them
		http://nlp.stanford.edu/data/glove.6B.zip
  - Change the desired configuration file in _/configs_ and replace all
		_<path-to>_ placeholders
  - run ```python run_experiment <path-to>/config.yaml```


## Visualization and End2End QA

For analysis and to visualize the attention weights we used our framework described 
in the demo paper "End-to-End Non-Factoid Question Answering with an Interactive 
Visualization of Neural Attention Weights" (Rücklé and Gurevych, ACL 2017). 
[The source code is available here.](https://github.com/UKPLab/acl2017-non-factoid-qa/blob/master/Candidate-Ranking/experiment/qa/model/lw_bilstm.py)
