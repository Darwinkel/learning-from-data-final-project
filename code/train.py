"""Train a model on a given dataset and save it to a file."""

import argparse
import json

from keras.callbacks import EarlyStopping
from utils.helpers import read_corpus
from utils.models import get_model


def create_arg_parser() -> argparse.Namespace:
    """Return parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Model to use")
    parser.add_argument("-t", "--train_file", type=str, help="File containing training data")
    parser.add_argument("-d", "--dev_file", type=str, help="File containing development data")
    parser.add_argument("-p", "--parameters", type=str, help="File containing hyperparameter configuration")
    parser.add_argument("-o", "--output_file", type=str, help="Location where trained model is saved")
    parser.add_argument("-e", "--embedding_file", type=str, help="File containing pretrained word embeddings")
    parser.add_argument("-hf", "--huggingface_model", type=str, help="Name of the model on HuggingFace")
    parser.add_argument("-ep", "--epochs", type=int, help="Number of epochs to train for")
    parser.add_argument("-bs", "--batch_size", type=int, help="Batch size")

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
    elif args.model == "lm":
        options = {"src": args.huggingface_model}
    else:
        options = {}

    if args.parameters:
        config = json.load(open(args.parameters))
        model = get_model(args.model, config=config, **options)
    else:
        model = get_model(args.model, **options)

    model.fit(
        X_train,
        y_train,
        X_dev=X_dev,
        y_dev=y_dev,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
    )

    model.save(args.output_file)


main()
