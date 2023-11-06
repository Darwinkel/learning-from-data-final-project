"""Helper functions for the project, which do not depend on heavy imports such as Tensorflow."""

import json

import numpy as np


def read_corpus(corpus_file: str) -> tuple[list[str], list[str]]:
    """Retrieve tweets and their offensiveness labels from a data file."""
    texts = []
    labels = []
    with open(corpus_file, encoding="utf-8") as file:
        for line in file:
            tokens = line.strip().split("\t")
            texts.append(tokens[0])
            labels.append(tokens[1])
    return texts, labels


def labellist2id(label_list: list[str]) -> np.array:
    """Convert list of labels to numpy array of binary ints."""
    return np.fromiter(map(label2id, label_list), dtype=int)


def id2label(label_id: int) -> str:
    """Convert label id to label string."""
    if label_id == 0:
        return "NOT"
    return "OFF"


def label2id(label: str) -> int:
    """Convert label string to label id."""
    if label == "NOT":
        return 0
    return 1


def read_embeddings(embeddings_file: str) -> dict[str, list[float]]:
    """Read in word embeddings from file and save as numpy array."""
    embeddings = json.load(open(embeddings_file))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc: list[str], emb: dict[str, list[float]]) -> np.array:
    """Get embedding matrix given vocab and the embeddings."""
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix
