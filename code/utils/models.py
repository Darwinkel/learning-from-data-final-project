"""Models definitions for the project, which depend on heavy imports such as Tensorflow."""

from functools import partial
from pathlib import Path

import keras_tuner
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import LSTM, Bidirectional, Dense, Embedding, TextVectorization
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import PredefinedSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from skops.io import dump, load
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from utils.helpers import get_emb_matrix, id2label, labellist2id, read_embeddings

optimizers = {
    "adam": Adam,
}

classifiers = {
    "SVC": LinearSVC,
    "SVM": lambda: SVC(kernel="rbf"),
    "DT": DecisionTreeClassifier,
    "RF": RandomForestClassifier,
}

vectorizers = {
    "count": CountVectorizer,
    "tfidf": TfidfVectorizer,
}


class AbstractModel:

    """Base class for all models."""

    def fit(self, X_train: list[str], y_train: list[str], **kwargs) -> None:
        """Train the model."""
        raise NotImplementedError

    def predict(self, X: list[str]) -> list[str]:
        """Output predictions."""
        raise NotImplementedError

    def save(self, filename: str) -> None:
        """Save model for later use."""
        raise NotImplementedError


class ClassicalModel(AbstractModel):

    """Classical SVM model."""

    def __init__(self, **kwargs) -> None:
        """Initialize model."""
        src = kwargs.get("src")
        vectorizer = kwargs.get("vectorizer")
        classifier = kwargs.get("classifier")
        if src:
            self.model = load(src, trusted=True)
        else:
            self.model = Pipeline(
                [
                    ("vect", vectorizer),
                    ("clf", classifier),
                ],
            )

    def fit(self, X_train: list[str], y_train: list[str], **kwargs) -> None:
        """Train the model."""
        self.model = self.model.fit(X_train, y_train)

    def predict(self, X: list[str]) -> list[str]:
        """Output predictions."""
        return self.model.predict(X)

    def save(self, filename: str) -> None:
        """Save model for later use."""
        dump(self.model, filename + ".classical")

    def score(self, *args, **kwargs) -> float:
        """Return model score."""
        return self.model.score(*args, **kwargs)

    def interpret(self) -> None:
        """Calculate feature importance."""
        n = 50
        vect = self.model.named_steps["vect"]
        clf = self.model.named_steps["clf"]
        for i, label in enumerate(clf.classes_):
            rest = [ind for ind, c in enumerate(clf.classes_) if ind != i]
            diff = [np.subtract(clf.feature_log_prob_[i, :], clf.feature_log_prob_[j, :]) for j in rest]
            sorted_ids = np.array(diff).argsort(axis=1)[:, ::-1]
            best_ids = sorted_ids[:, :n]
            best_names = np.take(vect.get_feature_names_out(), sorted_ids[:, :n])
            best_diff = np.exp(np.take(diff, sorted_ids[:, :n]))
            for j, c in enumerate(rest):
                print(f"{label} vs. {clf.classes_[c]}")
                for identifier, name, d in zip(best_ids[j], best_names[j], best_diff[j]):
                    print(
                        f"  {name} - ratio: {round(d, 2)} ({round(clf.feature_log_prob_[i, identifier], 2)} | {round(clf.feature_log_prob_[c, identifier], 2)})",
                    )


class LSTMModel(AbstractModel):

    """LSTM model."""

    def __init__(self, **kwargs) -> None:
        """Initialize the model."""
        src = kwargs.get("src")
        obj_src = kwargs.get("obj_src")
        vocab = kwargs.get("vocabulary")
        emb_file = kwargs.get("embedding_file")
        num_labels = kwargs.get("num_labels")
        if src:
            self.model = tf.keras.saving.load_model(src)
        elif obj_src:
            self.model = obj_src
        else:
            n_layers = kwargs.get("n_layers")
            bidi = kwargs.get("bidi")
            dense = kwargs.get("dense")
            trainable_emb = kwargs.get("trainable_emb")
            lr = kwargs.get("lr")
            loss = kwargs.get("loss")
            optimizer = kwargs.get("optimizer")
            dropout = kwargs.get("dropout")
            rec_dropout = kwargs.get("rec_dropout")
            activation = kwargs.get("activation")
            rec_activation = kwargs.get("rec_activation")

            # Prepare vectorizer
            vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
            # Adapt vectorizer to data vocabulary
            text_ds = tf.data.Dataset.from_tensor_slices(vocab)
            vectorizer.adapt(text_ds)
            wordids = vectorizer.get_vocabulary()

            # Prepare embeddings
            embeddings = read_embeddings(emb_file)
            emb_matrix = get_emb_matrix(wordids, embeddings)
            embedding_dim = len(emb_matrix[0])
            num_tokens = len(emb_matrix)

            # Construct model
            optim = optimizers[optimizer](learning_rate=lr)
            model = Sequential()
            model.add(vectorizer)
            model.add(
                Embedding(
                    num_tokens,
                    embedding_dim,
                    embeddings_initializer=Constant(emb_matrix),
                    trainable=trainable_emb,
                ),
            )
            if dense:
                model.add(Dense(300, activation=activation))
            for i in range(n_layers):
                lstm = LSTM(
                    300,
                    return_sequences=(i != (n_layers - 1)),
                    dropout=dropout,
                    recurrent_dropout=rec_dropout,
                    activation=activation,
                    recurrent_activation=rec_activation,
                )
                if bidi:
                    model.add(Bidirectional(lstm))
                else:
                    model.add(lstm)
            model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="softmax"))
            model.compile(loss=loss, optimizer=optim, metrics=["accuracy"])
            self.model = model

    def fit(self, X_train: list[str], y_train: list[str], X_dev: list[str], y_dev: list[str], **kwargs) -> None:
        """Train the model."""
        # Binarize labels
        labels = {
            "NOT": [1, 0],
            "OFF": [0, 1],
        }
        y_train_bin = np.array([labels[y] for y in y_train])
        y_dev_bin = np.array([labels[y] for y in y_dev])

        X_train = np.array([[s] for s in X_train])
        X_dev = np.array([[s] for s in X_dev])

        return self.model.fit(X_train, y_train_bin, validation_data=(X_dev, y_dev_bin), **kwargs)

    def predict(self, X: list[str]) -> list[str]:
        """Output predictions."""
        y_pred = self.model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        return map(id2label, y_pred)

    def save(self, filename: str) -> None:
        """Save model for later use."""
        tf.keras.saving.save_model(self.model, filename + ".lstm")

    def extract_model(self) -> tf.keras.Model:
        """Return the model itself."""
        return self.model


class LMModel(AbstractModel):

    """LM model."""

    def __init__(self, **kwargs) -> None:
        """Initialize the model."""
        src = kwargs.get("src")  # Path to the huggingface thing, can be local directory or name of pretrained model
        if not src:
            src = "distilbert-base-cased"

        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            src,
            num_labels=2,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(src)

    def fit(self, X_train: list[str], y_train: list[str], X_dev: list[str], y_dev: list[str], **kwargs) -> None:
        """Train the model."""
        tokens_train = self.tokenizer(X_train, padding=True, truncation=True, return_tensors="np").data
        tokens_dev = self.tokenizer(
            X_dev,
            padding=True,
            truncation=True,
            return_tensors="np",
        ).data

        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        y_train_bin = labellist2id(y_train)
        y_dev_bin = labellist2id(y_dev)

        self.model.fit(
            x=tokens_train,
            y=y_train_bin,
            validation_data=(tokens_dev, y_dev_bin),
            epochs=10,
            batch_size=16,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)],
            verbose=1,
        )

    def predict(self, X: list[str]) -> list[str]:
        """Output predictions."""
        tokens_test = self.tokenizer(X, padding=True, truncation=True, return_tensors="np").data
        Y_pred = self.model.predict(tokens_test)["logits"]
        # Finally, convert to numerical labels to get scores with sklearn
        Y_pred = np.argmax(Y_pred, axis=1)
        return map(id2label, Y_pred)

    def save(self, filename: str) -> None:
        """Save model and tokenizer for later use."""
        self.model.save_pretrained(filename + ".lm")
        self.tokenizer.save_pretrained(filename + ".lm")


def model_config(model_name: str, **kwargs) -> dict[str, AbstractModel]:
    """Return model configuration."""
    configs = {
        "baseline": partial(
            ClassicalModel,
            vectorizer=CountVectorizer(ngram_range=(1, kwargs.get("max_ngram", 3))),
            classifier=MultinomialNB(),
        ),
        "classical": partial(
            ClassicalModel,
            vectorizer=TfidfVectorizer(ngram_range=(1, kwargs.get("max_ngram", 1))),
            classifier=LinearSVC(),
        ),
        "lstm": LSTMModel,
        "lm": LMModel,
    }
    return configs[model_name]


def build_model(hp, **kwargs) -> AbstractModel:
    """Build model."""
    model_type = hp.Choice("model_type", ["baseline", "classical", "lstm", "lm"])
    if model_type == "baseline":
        model = ClassicalModel(
            vectorizer=CountVectorizer(
                analyzer=hp.Choice("analyzer", ["word", "char_wb"], default="word"),
                ngram_range=(1, hp.Int("ngram_max", min_value=1, max_value=5, step=1, default=3)),
            ),
            classifier=MultinomialNB(),
        )
    elif model_type == "classical":
        model = ClassicalModel(
            vectorizer=vectorizers[hp.Choice("vectorizer", ["count", "tfidf"], default="tfidf")](
                analyzer=hp.Choice("analyzer", ["word", "char_wb"], default="word"),
                ngram_range=(1, hp.Int("ngram_max", min_value=1, max_value=5, step=1, default=3)),
            ),
            classifier=classifiers[hp.Choice("classifier", ["SVC", "SVM", "DT", "RF"], default="SVC")](),
        )
    elif model_type == "lstm":
        model = LSTMModel(
            n_layers=hp.Int("n_layers", min_value=1, max_value=3, default=1),
            bidi=hp.Boolean("bidi", default=False),
            dense=hp.Boolean("dense", default=False),
            trainable_emb=hp.Boolean("trainable_embeddings", default=False),
            lr=hp.Float("learning_rate", min_value=1e-4, max_value=1e-1, step=10, sampling="log"),
            loss=hp.Choice("loss_function", ["categorical_crossentropy", "mse"], default="categorical_crossentropy"),
            optimizer=hp.Choice("optimizer", ["adam"], default="adam"),
            dropout=hp.Float("input_dropout", min_value=0, max_value=0.2, step=0.05),
            rec_dropout=hp.Float("recurrent_dropout", min_value=0, max_value=0.2, step=0.05),
            activation=hp.Choice("input_activation", ["tanh", "sigmoid"], default="sigmoid"),
            rec_activation=hp.Choice("recurrent_activation", ["tanh", "sigmoid"], default="sigmoid"),
            **kwargs,
        )
    elif model_type == "lm":
        model = LMModel(**kwargs)
    return model


class MyHyperModel(keras_tuner.HyperModel):

    """HyperModel for Bayesian optimization."""

    def __init__(self, **kwargs) -> None:
        """Initialize the hypermodel."""
        self.params = kwargs

    def build(self, hp) -> AbstractModel:
        """Build the hypermodel."""
        model = build_model(hp, **self.params)
        self.hp = hp
        return model.extract_model()

    def fit(self, hp, model, *args, **kwargs) -> None:
        """Fit the hypermodel."""
        model = build_model(self.hp, obj_src=model)
        X_train = args[0]
        y_train = args[1]
        validation_data = kwargs.get("validation_data")
        X_dev = validation_data[0]
        y_dev = validation_data[1]
        del kwargs["validation_data"]
        return model.fit(X_train, y_train, X_dev, y_dev, **kwargs)


def get_model(model_type: str, **kwargs) -> AbstractModel:
    """Return model."""
    config = kwargs.get("config")
    if config:
        hp = keras_tuner.HyperParameters.from_config(config)
    else:
        hp = keras_tuner.HyperParameters()
        hp.Fixed("model_type", model_type)
    return build_model(hp, **kwargs)


def load_model(filename: Path) -> AbstractModel:
    """Load model from disk."""
    model_type = Path(filename).suffix[1:]
    models = {
        "classical": ClassicalModel,
        "lstm": LSTMModel,
        "lm": LMModel,
    }
    return models[model_type](src=filename)


def tune_model(
    model_type: str,
    max_trials: int,
    X_train,
    y_train,
    X_dev,
    y_dev,
    project_name="search",
    **kwargs,
) -> list:
    """Tune hypermodel."""
    hp = keras_tuner.HyperParameters()
    hp.Fixed("model_type", model_type)
    hypermodel = partial(build_model, **kwargs)
    if model_type in ("classical", "baseline"):
        # Sklearn Tuner doesn't accept validation data explicitly
        # Merge train and dev data, and use indices to split back into train and test during search
        X_combined = X_train + X_dev
        y_combined = y_train + y_dev
        split_ids = np.repeat([-1, 0], [len(y_train), len(y_dev)])
        ps = PredefinedSplit(split_ids)
        tuner = keras_tuner.SklearnTuner(
            oracle=keras_tuner.oracles.GridSearchOracle(
                objective=keras_tuner.Objective("score", "max"),
                max_trials=max_trials,
                hyperparameters=hp,
            ),
            hypermodel=hypermodel,
            cv=ps,
            overwrite=True,
        )
        tuner.search(np.array(X_combined), np.array(y_combined))
    else:
        tuner = keras_tuner.BayesianOptimization(
            MyHyperModel(**kwargs),
            objective="val_accuracy",
            max_trials=max_trials,
            hyperparameters=hp,
            project_name=project_name,
            overwrite=True,
        )
        tuner.search(
            X_train,
            y_train,
            validation_data=(X_dev, y_dev),
            epochs=50,
            batch_size=32,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
        )
    return tuner.get_best_hyperparameters()
