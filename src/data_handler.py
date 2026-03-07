from collections import Counter
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_text as tf_text
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

PAD = "<pad>"
UNK = "<unk>"
MAX_LEN = 64
BATCH_SIZE = 64
TRAIN_SIZE = 5000
TEST_SIZE = 500

def get_data(
    path: str,
) -> tuple[
    Any,  # X_train
    Any,  # X_validation
    Any,  # X_test
    Any,  # y_train
    Any,  # y_validation
    Any,  # y_test
    Any,  # original_train
    Any,  # original_validation
    Any,  # original_test
]:
    """
    Load the raw data from the path and split it into training, validation, and test data.
    Returns the processed features and labels, and the original data.

    Arguments:
        path: str- The path to the dataset.

    Returns:
        Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]
            A tuple which contains:
            - X_train: Processed training data.
            - X_validation: Processed validation data.
            - X_test: Processed test data.
            - y_train: Training labels.
            - y_validation: Validation labels.
            - y_test: Test labels.
            - original_train: Original training data.
            - original_validation: Original validation data.
            - original_test: Original test data.
    """
    training_data, validation_data, test_data, y_train, y_validation, y_test = (
        read_data(path)
    )
    (
        X_train,
        X_validation,
        X_test,
        original_train,
        original_validation,
        original_test,
        vocab,
    ) = preprocess_data(training_data, validation_data, test_data, y_train,y_validation,y_test)

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        original_train,
        original_validation,
        original_test,
        vocab,
    )


def read_data(
    path: str,
) -> tuple[
    pd.DataFrame,  # training data
    pd.DataFrame,  # validation data
    pd.DataFrame,  # test data
    pd.Series,  # y_train
    pd.Series,  # y_validation
    pd.Series,  # y_test
]:
    """
    Function reads the data from train.csv and test.csv, and extracts the
    column Class Index as labels; Splits the training data into training
    and validation, then prints the lengths of all data splits.

    Arguments:
        path: str- The path to the dataset.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]
            A tuple which contains:
            - training_data: Training data.
            - validation_data: Validation data.
            - test_data: Test data.
            - y_train: Training labels.
            - y_validation: Validation labels.
            - y_test: Test labels.
    """
    train_data = pd.read_csv(path + "/train.csv")[:TRAIN_SIZE]
    test_data = pd.read_csv(path + "/test.csv")[:TEST_SIZE]

    labels_train = train_data["Class Index"]
    y_test = test_data["Class Index"]

    training_data, validation_data, y_train, y_validation = train_test_split(
        train_data, labels_train, test_size=0.1, random_state=67
    )

    print(f"length of train: {len(training_data)}")
    print(f"length of labels train: {len(y_train)}")
    print(f"length of validation: {len(validation_data)}")
    print(f"length of labels validation: {len(y_validation)}")
    print(f"length of test: {len(test_data)}")
    print(f"length of test labels: {len(y_test)}")

    return training_data, validation_data, test_data, y_train, y_validation, y_test


def tokenize_data(data):
    tokenizer = tf_text.UnicodeScriptTokenizer()
    tokens = tokenizer.tokenize([data])
    return tokens.to_list()[0]


def build_vocab(texts, min_freq: int = 2, max_size: int = 30000) -> dict:
    """
    Build a vocabulary mapping from tokens to integer indices.
    The vocabulary will include only tokens that appear at least `min_freq` times,
    and will be limited to `max_size` tokens (including PAD and UNK).
    """
    counter = Counter()
    for text in texts:
        counter.update(tokenize_data(text))
    # Reserve 0 for PAD and 1 for UNK.
    vocab = {PAD: 0, UNK: 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab


def numericalize(tokens: list, vocab: dict) -> list:
    """
    Convert a list of tokens into a list of integer indices using the provided vocabulary.
    Tokens not found in the vocabulary will be mapped to the index of UNK.
    """
    return [vocab.get(tok, vocab[UNK]) for tok in tokens]


@dataclass
class Batch:
    x: torch.Tensor  # (B, T) token ids
    lengths: torch.Tensor  # (B,) true lengths
    y: torch.Tensor  # (B,) labels


class TextDataset(Dataset):
    def __init__(self, data, labels, vocab: dict, max_len: int = 200) -> None:
        self.data = data
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        """Given an index, return the token ids and label for the corresponding sample."""
        item = self.data[idx]
        tokens = tokenize_data(item)

        # Convert to ids and truncate
        if len(tokens) == 0:
            ids = [self.vocab[UNK]]
        else:
            ids = numericalize(tokens, self.vocab)[: self.max_len]
            if len(ids) == 0:
                ids = [self.vocab[UNK]]
        return ids, int(self.labels[idx])


def collate(batch: list) -> Batch:
    """Collate function to convert a list of samples into a batch."""
    # batch: list of (ids_list, label)
    lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(batch) > 0 else 0
    x = torch.full((len(batch), max_len), 0, dtype=torch.long)
    y = torch.tensor([y for _, y in batch], dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return Batch(x=x, lengths=lengths, y=y)


def plot_lengths(data):
    Path("plots").mkdir(exist_ok=True)
    
    lengths = [len(tokenize_data(text)) for text in data]
    plt.hist(lengths, bins=50)
    plt.title("Distribution of tokenized text lengths in training set")
    plt.xlabel("Length of tokenized text")
    plt.ylabel("Frequency")
    plt.savefig("plots/lengths_distribution")
    plt.close()


def preprocess_data(
    training_data: pd.DataFrame, validation_data: pd.DataFrame, test_data: pd.DataFrame,  y_train,y_validation,y_test
) -> tuple[
    Any,  # X_train
    Any,  # X_validation
    Any,  # X_test
    pd.Series,  # original_train
    pd.Series,  # original_validation
    pd.Series,  # original_test
]:
    """
    Function combines the Title an Description columns into one variable, and
    applies a TF- IDF vectorizer (with english stopwords). Then, the vectorizer
    is used to transform the training, validation, and test data sets.

    Arguments:
        training_data: pd.DataFrame- Training data which contains the Title
            and Description columns.
        validation_data: pd.DataFrame- Validation data which contains the Title
            and Description columns.
        test_data: pd.DataFrame- Test data which contains the Title
            and Description columns.
    Returns:
        Tuple[csr_matrix, csr_matrix, csr_matrix, Series, Series, Series]
            A tuple which contains:
            - X_train: TF-IDF training set.
            - X_validation: TF-IDF validation set.
            - X_test: TF-IDF test set.
            - original_train: Training set.
            - original_validation: Validation set.
            - original_test: Test set.
    """
    original_train = (training_data["Title"] + training_data["Description"]).values.tolist()
    original_validation = (validation_data["Title"] + validation_data["Description"]).values.tolist()
    original_test = (test_data["Title"] + test_data["Description"]).values.tolist()

    # tokenisation
    vocab = build_vocab(original_train, min_freq=2, max_size=30000)
    
    # # Plot distribution of lengths in the training set
    # plot_lengths(original_train)

    print(f"Using MAX_LEN={MAX_LEN} and BATCH_SIZE={BATCH_SIZE}")
    train_ds = TextDataset(original_train, y_train.values.tolist(), vocab, max_len=MAX_LEN)
    val_ds = TextDataset(original_validation, y_validation.values.tolist(), vocab, max_len=MAX_LEN)
    test_ds = TextDataset(original_test, y_test.values.tolist(), vocab, max_len=MAX_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        original_train,
        original_validation,
        original_test,
        vocab,
    )
