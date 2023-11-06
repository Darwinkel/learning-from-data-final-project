format:
	ruff .
	black .
	mypy .

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

copy:
	rsync -avx ./ snum@login1.hb.hpc.rug.nl:~/lfd/final-project

ssh:
	ssh snum@login1.hb.hpc.rug.nl

gpu-shell:
	srun --gpus-per-node=1 --time=04:00:00 --pty /bin/bash

queue-translations:
	ssh -t snum@login1.hb.hpc.rug.nl  'cd lfd/final-project && sbatch scripts/jobscript_translation_train_nl.sh && sbatch scripts/jobscript_translation_dev_nl.sh && sbatch scripts/jobscript_translation_test_nl.sh && sbatch scripts/jobscript_translation_train_he.sh && sbatch scripts/jobscript_translation_dev_he.sh && sbatch scripts/jobscript_translation_test_he.sh'

queue-train-predict:
	ssh -t snum@login1.hb.hpc.rug.nl  'cd lfd/final-project && sbatch scripts/jobscript_train_predict_en_en.sh && sbatch scripts/jobscript_train_predict_en_nl-he.sh  && sbatch scripts/jobscript_train_predict_he_he.sh  && sbatch scripts/jobscript_train_predict_nl_nl.sh && sbatch scripts/jobscript_train_predict_multi_all.sh && sbatch scripts/jobscript_train_predict_multi_nl-he.sh'

run-evaluation:
	# General classification scores - mono-mono
	python3 code/evaluate.py -t data/test.tsv -p predictions/pred-roberta-base-en-en.tsv
	python3 code/evaluate.py -t data/test-he.tsv -p predictions/pred-hero-he-he.tsv
	python3 code/evaluate.py -t data/test-nl.tsv -p predictions/pred-robbert-nl-nl.tsv

	# General classification scores - multi-all
	python3 code/evaluate.py -t data/test-all.tsv -p predictions/pred-xlm-roberta-base-multi-all.tsv

	# General classification scores - multi-{en,he,nl}
	python3 code/evaluate.py -t data/test.tsv -p predictions/pred-xlm-roberta-base-en-en.tsv
	python3 code/evaluate.py -t data/test-he.tsv -p predictions/pred-xlm-roberta-base-he-he.tsv
	python3 code/evaluate.py -t data/test-nl.tsv -p predictions/pred-xlm-roberta-base-nl-nl.tsv

	# General classification scores - multi-nl-he vs mono-nl-he
	python3 code/evaluate.py -t data/test-nl-he.tsv -p predictions/pred-xlm-roberta-base-multi-nl-he.tsv
	python3 code/evaluate.py -t data/test-nl-he.tsv -p predictions/pred-roberta-base-en-nl-he.tsv

	# Agreement between models - mono-mono
	python3 code/evaluate.py -c predictions/pred-roberta-base-en-en.tsv -p predictions/pred-hero-he-he.tsv
	python3 code/evaluate.py -c predictions/pred-robbert-nl-nl.tsv -p predictions/pred-roberta-base-en-en.tsv
	python3 code/evaluate.py -c predictions/pred-robbert-nl-nl.tsv -p predictions/pred-hero-he-he.tsv

	# Agreement between models - multi-mono
	python3 code/evaluate.py -c predictions/pred-roberta-base-en-nl-he.tsv -p predictions/pred-xlm-roberta-base-multi-nl-he.tsv