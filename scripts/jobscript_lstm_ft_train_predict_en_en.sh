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

python3 code/train.py -m lstm -t data/train.tsv -d data/dev.tsv -p models/lstm_ft_best.json -o models/lstm_ft_best -e embeddings/ft-en.json
python3 code/predict.py -t lstm -m models/lstm_ft_best.lstm -i data/test.tsv -o predictions/lstm-ft-en-en.tsv

deactivat 