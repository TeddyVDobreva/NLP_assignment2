import random
import time

import numpy as np
import torch
import torch.nn as nn

from src.evaluation import (
    plot_learning_curves,
    plot_confucion_matrix,
    compare,
    do_final_evaluation,
    get_misclassified_examples,
)
from src.ablation import ablation
from src.data_handler import get_preprocessed_data, get_from_fast_file
from src.hyperparameter_evaluation import do_hyperparameter_evaluation
from src.models import CNNTextClassifier, LSTMClassifier


def train_and_time(name: str, model: nn.Module, train, validation):
    t0 = time.perf_counter()
    hist = model.fit(
        train,
        validation,
    )
    total_time = time.perf_counter() - t0
    val = model.evaluate(validation)
    return {
        "name": name,
        "hist": hist,
        "val": val,
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


# get the data
train_loader, val_loader, test_loader, vocab = get_preprocessed_data("data", plots=True)
# train_loader, val_loader, test_loader, vocab = get_from_fast_file()
vocab_size = len(vocab)

set_seed()

# Hyperparameter tuning
do_hyperparameter_evaluation(
    CNNTextClassifier,
    {"lr": [0.01, 0.001, 0.0001]},
    {"embed_dim": [64, 128, 256]},
    vocab_size=vocab_size,
    train_loader=train_loader,
    validation_loader=val_loader,
)

# initialize models with best hyperparameters
lstm = LSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=128,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3,
    pad_idx=0,
)

cnn = CNNTextClassifier(
    vocab_size=vocab_size,
    embed_dim=128,
    num_filters=64,
    kernel_sizes=(3, 4),
    dropout=0.3,
    pad_idx=0,
)

# train models
print("Training LSTM...")
res_lstm = train_and_time("LSTM", lstm, train_loader, val_loader)

print("Training CNN...")
res_cnn = train_and_time("CNN", cnn, train_loader, val_loader)

# print results from training
compare([res_lstm, res_cnn])

# plot learning curves
plot_learning_curves([res_lstm, res_cnn])

# plot confusion matrixes
plot_confucion_matrix(lstm, val_loader, "LSTM", "validation")
plot_confucion_matrix(cnn, val_loader, "CNN", "validation")

# do ablation study
print("\nAblation results")
ablation(
    LSTMClassifier,
    "LSTM",
    {"dropout": [0, 0.3]},
    train_loader,
    val_loader,
    vocab_size=vocab_size,
    embed_dim=128,
    hidden_dim=64,
    num_layers=2,
    pad_idx=0,
)
ablation(
    CNNTextClassifier,
    "CNN",
    {"dropout": [0, 0.3]},
    train_loader,
    val_loader,
    vocab_size=vocab_size,
    embed_dim=128,
    num_filters=64,
    kernel_sizes=(3, 4),
    pad_idx=0,
)

# do final evaluation
do_final_evaluation(lstm, test_loader, "LSTM", "testing")
do_final_evaluation(cnn, test_loader, "CNN", "testing")

# error analysis
get_misclassified_examples(lstm, "LSTM", 'data', vocab, max_items = 10)
get_misclassified_examples(cnn, "CNN", 'data', vocab, max_items = 10)