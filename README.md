# Learning from Data - Final Project

## Installation
See `requirements.txt` and `requirements-dev.txt`. The code targets Python 3.10 and 3.11.

## Data
The used datasets are in the `data` folder. The used FastText and GloVe files can be found in the `embeddings` folder.

## Trained models and results
Some of the models we trained can be found in the `models` folder. Predictions made by our models can be found in the `predictions` folder.

## Replicating our experiment
Overall, the `scripts` folder contains all the jobscripts needed to replicate our experiment on the Hábrók cluster (usable with `sbatch`). The `Makefile` contains some shortcuts and helpers to run the scripts.
Note that a virtual environment should be created beforehand. 

### Preprocessing the data
See `python code/preprocess.py --help`

### Translating the English source data
See `python code/translate.py --help` and `scripts/jobscript_translation*.sh`

### Training a model
See `python code/train.py --help` and `scripts/jobscript_*train_predict*.sh`

### Predicting with a model
See `python code/predict.py --help` and `scripts/jobscript_*train_predict*.sh`

### Evaluating a model
See `python code/evaluate.py --help` and `run-evaluation` in the `Makefile`.

### Hyperparameter search
See `python code/search.py --help`  and `scripts/jobscript_*search.sh`

### Interpreting model weights
See `python code/interpret.py --help` 