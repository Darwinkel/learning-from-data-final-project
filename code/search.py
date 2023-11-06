"""Search for the best hyperparameters for a model."""

import argparse
import json

from utils.helpers import read_corpus
from utils.models import tune_model


def create_arg_parser() -> argparse.Namespace:
    """Return parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Model to use")
    parser.add_argument("-t", "--train_file", type=str, help="File containing training data")
    parser.add_argument("-d", "--dev_file", type=str, help="File containing development data")
    parser.add_argument("-o", "--output_file", type=str, help="Location where trained model is saved")
    parser.add_argument("-e", "--embedding_file", type=str, help="File containing pretrained word embeddings")
    parser.add_argument("-mt", "--max_trials", type=int, default=10, help="Maximum number of search trials")
    parser.add_argument("-n", "--name", type=str, help="Name of search project")

    return parser.parse_args()


def main() -> None:
    """Run main function."""
    args = create_arg_parser()
    X_train, y_train = read_corpus(args.train_file)
    X_dev, y_dev = read_corpus(args.dev_file)

    if args.model == "lstm":
        options = {
            "embedding_file": args.embedding_file,
            "vocabulary": X_train + X_dev,
            "num_labels": len(set(y_train)),
        }
    else:
        options = {}

    best_hp = tune_model(
        args.model,
        args.max_trials,
        X_train,
        y_train,
        X_dev=X_dev,
        y_dev=y_dev,
        project_name=args.name,
        **options,
    )
    json.dump(best_hp[0].get_config(), open(args.output_file, "w"))


main()
