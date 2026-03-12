import random
import time

import numpy as np
import torch
import torch.nn as nn
from pandas import DataFrame

from src.functions_models import evaluation_loop, training_loop
from src.hyperparameter_evaluation import do_hyperparameter_evaluation, make_heatmap
from src.data_handler import get_preprocessed_data
from src.models import CNNTextClassifier, LSTMClassifier

MAX_EPOCHS = 12
PATIENCE = 3
LR = 1e-3

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_time(name: str, model: nn.Module):
    t0 = time.perf_counter()
    hist = training_loop(
        model,
        train_loader,
        val_loader,
        lr=LR,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
    )
    total_time = time.perf_counter() - t0
    val = evaluation_loop(model, val_loader)
    test = evaluation_loop(model, test_loader)
    return {
        "name": name,
        "hist": hist,
        "val": val,
        "test": test,
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


train_loader, val_loader, test_loader, vocab = get_preprocessed_data('data', True)
vocab_size = len(vocab)

set_seed(67)

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

print("Number of trainable parameters:")
print("LSTM:", count_parameters(lstm))
print("CNN: ", count_parameters(cnn))

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
