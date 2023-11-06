#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --mem=8000

module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 SciPy-bundle scikit-learn

source $HOME/venvs/lfd_fp/bin/activate

pip install --upgrade pip wheel
pip install transformers pandas

python3 --version
which python3

python3 code/translate.py -i data/test.tsv -o data/test-nl.tsv -l nl

deactivate
