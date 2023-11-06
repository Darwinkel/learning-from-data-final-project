"""Filter embeddings to only include words in the vocabulary."""

import argparse
import json

import numpy as np
from tqdm import tqdm
from utils.helpers import read_corpus


def create_arg_parser() -> argparse.Namespace:
    """Return parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_file", type=str, help="Name of file containing embeddings")
    parser.add_argument("-o", "--output_file", help="Name of file where filtered embedding dictionary will be saved")
    parser.add_argument("-v", "--vocabulary", nargs="+", help="Files containing text")
    return parser.parse_args()


def main() -> None:
    """Run main function."""
    args = create_arg_parser()
    vocab = []
    for file in args.vocabulary:
        text, labels = read_corpus(file)
        for item in text:
            vocab.extend(item.split(" "))
    embeddings = {}
    num_lines = sum(1 for line in open(args.base_file, encoding="utf-8"))
    with open(args.base_file, encoding="utf-8") as file:
        for line in tqdm(file, total=num_lines):
            tokens = line.split(" ", maxsplit=1)
            word = tokens[0]
            if word in vocab:
                vec = list(np.asarray(tokens[1].split(" "), dtype=float))
                embeddings[word] = vec
    json.dump(embeddings, open(args.output_file, "w"))


main()
