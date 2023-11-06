#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16000

module purge
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0 SciPy-bundle scikit-learn

source $HOME/venvs/lfd_fp/bin/activate

pip install --upgrade pip wheel
pip install transformers pandas keras keras_tuner skops tensorflow

python3 --version
which python3

python3 code/train.py -m lm -t data/train-nl-he.tsv -d data/dev-nl-he.tsv -hf roberta-base -o models/roberta-base-en-nl-he
python3 code/predict.py -t lm -m models/roberta-base-en-nl-he.lm -i data/test-nl-he.tsv -o predictions/pred-roberta-base-en-nl-he.tsv

deactivate
