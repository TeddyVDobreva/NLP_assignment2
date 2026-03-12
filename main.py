import random
import time

import numpy as np
import torch
import torch.nn as nn
from pandas import DataFrame

from src.data_handler import get_preprocessed_data
from src.hyperparameter_evaluation import do_hyperparameter_evaluation
from src.models import CNNTextClassifier, LSTMClassifier


def train_and_time(name: str, model: nn.Module):
    t0 = time.perf_counter()
    hist = model.fit(
        train_loader,
        val_loader,
    )
    total_time = time.perf_counter() - t0
    val = model.evaluate(val_loader)
    return {
        "name": name,
        "hist": hist,
        "val": val,
        # "test": test,
        "time_s_total": total_time,
    }


def set_seed(seed: int = 67) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


train_loader, val_loader, test_loader, vocab = get_preprocessed_data("data", False, True)
vocab_size = len(vocab)

set_seed()

lstm = LSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=64,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3,
    pad_idx=0,
)

cnn = CNNTextClassifier(
    vocab_size=vocab_size,
    embed_dim=64,
    num_filters=64,
    kernel_sizes=(3, 4, 5),
    dropout=0.3,
    pad_idx=0,
)

# Hyperparameter tuning
do_hyperparameter_evaluation(
    CNNTextClassifier,
    {"lr": [0.01, 0.001, 0.0001]},
    {"embed_dim": [64, 128, 256]},
    vocab_size=vocab_size,
    train_loader=train_loader,
    validation_loader=val_loader,
)

print("Training LSTM...")
res_lstm = train_and_time("LSTM", lstm)

print("Training CNN...")
res_cnn = train_and_time("CNN", cnn)

rows = []
for res in [res_lstm, res_cnn]:
    rows.append(
        [
            res["name"],
            res["val"]["acc"],
            res["val"]["f1"],
            res["test"]["acc"],
            res["test"]["f1"],
            res["time_s_total"],
        ]
    )

df_compare = (
    DataFrame(
        rows,
        columns=[
            "model",
            "val_acc",
            "val_macro_f1",
            "test_acc",
            "test_macro_f1",
            "train_time_s",
        ],
    )
    .sort_values(by=["val_macro_f1", "val_acc"], ascending=False)
    .reset_index(drop=True)
)

print(df_compare)
