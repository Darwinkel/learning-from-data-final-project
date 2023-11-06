#!/bin/bash

python code/search.py -m classical -t data/train.tsv -d data/dev.tsv -o models/svm-best.json
python code/train.py -p models/svm-best.json -t data/train.tsv -d data/dev.tsv -o models/svm-best
python code/predict.py -m models/svm-best.classical -i data/test.tsv -o models/svm-1.tsv
python code/evaluate.py -t data/test.tsv -p models/svm-1.tsv
python code/interpret.py -m models/svm-best.classical