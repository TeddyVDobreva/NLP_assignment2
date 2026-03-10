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
from datasets import Dataset as ds

PAD = "<pad>"
UNK = "<unk>"
MAX_LEN = 64
BATCH_SIZE = 32

# Subsample for speed
N_TRAIN = 512
N_VAL = 128
N_TEST = 128


def get_raw_data(path):
    train_data = pd.read_csv(path + "/train.csv")
    test_data = pd.read_csv(path + "/test.csv")

    train_data, validation_data = train_test_split(
        train_data, test_size=0.1, random_state=67
    )

    X_train = pd.DataFrame(
        {
            "text": train_data["Title"] + train_data["Description"],
            "label": train_data["Class Index"],
        }
    )
    X_validation = pd.DataFrame(
        {
            "text": validation_data["Title"] + validation_data["Description"],
            "label": validation_data["Class Index"],
        }
    )
    X_test = pd.DataFrame(
        {
            "text": test_data["Title"] + test_data["Description"],
            "label": test_data["Class Index"]
        }
    )

    return {
        "train": ds.from_pandas(X_train, preserve_index=False),
        "validation": ds.from_pandas(X_validation, preserve_index=False),
        "test": ds.from_pandas(X_test, preserve_index=False),
    }


def get_smaller_datasets(raw):
    train = raw["train"].shuffle(seed=67).select(range(N_TRAIN))
    validation = raw["validation"].shuffle(seed=67).select(range(N_VAL))
    test = raw["test"].select(range(N_TEST))
    return train, validation, test


def get_datasets(raw):
    train = raw["train"].shuffle(seed=67)  # We only shuffle the training set
    validation = raw["validation"]
    test = raw["test"]
    return train, validation, test


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
    def __init__(self, hf_ds: dict, vocab: dict, max_len: int = 200) -> None:
        self.ds = hf_ds
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple:
        """Given an index, return the token ids and label for the corresponding sample."""
        item = self.ds[idx]
        tokens = tokenize_data(item["text"])

        # Convert to ids and truncate
        if len(tokens) == 0:
            ids = [self.vocab[UNK]]
        else:
            ids = numericalize(tokens, self.vocab)[: self.max_len]
            if len(ids) == 0:
                ids = [self.vocab[UNK]]

        label = int(item["label"]) -1  # 0 negative, 1 positive
        return ids, label


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

    lengths = [len(tokenize_data(text)) for text in data["text"]]
    plt.hist(lengths, bins=50)
    plt.title("Distribution of tokenized text lengths in training set")
    plt.xlabel("Length of tokenized text")
    plt.ylabel("Frequency")
    plt.savefig("plots/lengths_distribution")
    plt.close()


def preprocess_data(
    train_ds_hf: pd.DataFrame,
    val_ds_hf: pd.DataFrame,
    test_ds_hf: pd.DataFrame,
) -> tuple[
    Any,  # X_train
    Any,  # X_validation
    Any,  # X_test
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

    vocab = build_vocab(train_ds_hf["text"], min_freq=2, max_size=30000)

    print(f"Using MAX_LEN={MAX_LEN} and BATCH_SIZE={BATCH_SIZE}")
    
    sample = train_ds_hf[0]["text"]
    print(tokenize_data(sample)[:20], numericalize(tokenize_data(sample)[:20], vocab)[:20])

    train_ds = TextDataset(train_ds_hf, vocab, max_len=MAX_LEN)
    val_ds = TextDataset(val_ds_hf, vocab, max_len=MAX_LEN)
    test_ds = TextDataset(test_ds_hf, vocab, max_len=MAX_LEN)

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
        vocab,
    )
