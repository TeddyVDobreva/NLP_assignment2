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
import tensorflow as tf

PAD = "<pad>"
UNK = "<unk>"
MAX_LEN = 64
BATCH_SIZE = 64

# Subsample for speed
N_TRAIN = 500
N_VAL = 100
N_TEST = 100


def _get_raw_data(path):
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
            "label": test_data["Class Index"],
        }
    )

    return {
        "train": ds.from_pandas(X_train, preserve_index=False),
        "validation": ds.from_pandas(X_validation, preserve_index=False),
        "test": ds.from_pandas(X_test, preserve_index=False),
    }


def _get_smaller_datasets(raw):
    train = raw["train"].shuffle(seed=67).select(range(N_TRAIN))
    validation = raw["validation"].shuffle(seed=67).select(range(N_VAL))
    test = raw["test"].select(range(N_TEST))
    return train, validation, test


def _get_datasets(raw):
    train = raw["train"].shuffle(seed=67)  # We only shuffle the training set
    validation = raw["validation"]
    test = raw["test"]
    return train, validation, test


def _tokenize_data(data):
    tokenizer = tf_text.UnicodeScriptTokenizer()
    tokens = tokenizer.tokenize([data])
    return tokens.to_list()[0]


def _build_vocab(texts, min_freq: int = 2, max_size: int = 30000) -> dict:
    """
    Build a vocabulary mapping from tokens to integer indices.
    The vocabulary will include only tokens that appear at least `min_freq` times,
    and will be limited to `max_size` tokens (including PAD and UNK).
    """
    counter = Counter()
    for text in texts:
        counter.update(_tokenize_data(text))
    # Reserve 0 for PAD and 1 for UNK.
    vocab = {PAD: 0, UNK: 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab


def _numericalize(tokens: list, vocab: dict) -> list:
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
    def __init__(self, hf_ds: dict, vocab: dict, max_len: int = 200, from_file = False, nums = None, labs = None) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.labels = []
        self.numericalized = []
        if from_file:
            self.numericalized = nums
            self.labels = labs
        else:
            self._numericalize_all(hf_ds)
        
    def _numericalize_all(self, hf_ds):
        for item in hf_ds:
            tokens = _tokenize_data(item["text"])
            # Convert to ids and truncate
            if len(tokens) == 0:
                ids = [self.vocab[UNK]]
            else:
                ids = _numericalize(tokens, self.vocab)[: self.max_len]
                if len(ids) == 0:
                    ids = [self.vocab[UNK]]
            self.numericalized.append(ids)
            self.labels.append(int(item["label"]) -1)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        """Given an index, return the token ids and label for the corresponding sample."""
        return self.numericalized[idx][: self.max_len], self.labels[idx]


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


def _plot_lengths(data):
    Path("plots").mkdir(exist_ok=True)

    lengths = [len(_tokenize_data(text)) for text in data["text"]]
    plt.hist(lengths, bins=50)
    plt.title("Distribution of tokenized text lengths in training set")
    plt.xlabel("Length of tokenized text")
    plt.ylabel("Frequency")
    plt.savefig("plots/lengths_distribution")
    plt.close()


def get_preprocessed_data(
    path, small_datasets=False, plots=False
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
    raw = _get_raw_data(path)

    if small_datasets:
        train_ds_hf, val_ds_hf, test_ds_hf = _get_smaller_datasets(raw)
    else:
        train_ds_hf, val_ds_hf, test_ds_hf = _get_datasets(raw)

    print(
        f"Dataset lengths: train={len(train_ds_hf)}, val={len(val_ds_hf)}, test={len(test_ds_hf)}"
    )

    vocab = _build_vocab(train_ds_hf["text"], min_freq=2, max_size=30000)
    
    if plots:
        _plot_lengths(train_ds_hf)

    print(f"Using MAX_LEN={MAX_LEN} and BATCH_SIZE={BATCH_SIZE}")
  
    train_ds = TextDataset(train_ds_hf, vocab, max_len=MAX_LEN)
    print('check')
    val_ds = TextDataset(val_ds_hf, vocab, max_len=MAX_LEN)
    print('check')
    test_ds = TextDataset(test_ds_hf, vocab, max_len=MAX_LEN)
    print('check')
    
    with open("data/fast.txt", "a") as f:
        for j in vocab:
            try:
                word = str(j).split("'")[1]
            except IndexError:
                word = str(j)
            f.write(f"{word}, {vocab[j]}\n")
        f.write("\n")
        for i in range(len(train_ds)):
            for j in train_ds[i][0]:
                f.write(f"{j},")
            f.write(f"{train_ds[i][1]}\n")
        f.write("\n")
        for i in range(len(val_ds)):
            for j in val_ds[i][0]:
                f.write(f"{j},")
            f.write(f"{val_ds[i][1]}\n")
        f.write("\n")
        for i in range(len(test_ds)):
            for j in test_ds[i][0]:
                f.write(f"{j},")
            f.write(f"{test_ds[i][1]}\n")
        

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


def get_from_fast_file():
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    test_data = []
    test_labels = []
    vocab = {}
    with open('data/fast.txt', 'r') as file:
        for line in file:
            if line == "\n" or line is None:
                break
            t = line.strip().split(', ')
            vocab[t[0].encode() if t[1] not in ['0', '1'] else t[0]]=int(t[1])
        for line in file:
            if line == "\n" or line is None:
                break
            t = line.strip().split(',')
            train_data.append([int(x) for x in t[:-1]])
            train_labels.append(int(t[-1]))
        for line in file:
            if line == "\n" or line is None:
                break
            t = line.strip().split(',')
            val_data.append([int(x) for x in t[:-1]])
            val_labels.append(int(t[-1]))
        for line in file:
            t = line.strip().split(',')
            test_data.append([int(x) for x in t[:-1]])
            test_labels.append(int(t[-1]))
            
    train_ds = TextDataset({}, vocab, max_len=MAX_LEN, from_file=True, nums=train_data, labs=train_labels)
    val_ds = TextDataset({}, vocab, max_len=MAX_LEN, from_file=True, nums=val_data, labs=val_labels)
    test_ds = TextDataset({}, vocab, max_len=MAX_LEN, from_file=True, nums=test_data, labs=test_labels)
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
        
    