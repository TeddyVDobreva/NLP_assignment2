import random
import time

import numpy as np
import torch
import torch.nn as nn
from pandas import DataFrame

from functions_models import evaluation_loop, training_loop
from hyperparameter_evaluation import do_hyperparameter_evaluation, make_heatmap
from src.data_handler import get_raw_data, get_smaller_datasets, preprocess_data
from src.models import CNNTextClassifier, LSTMClassifier


def set_seed(seed: int = 67) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(13)

raw = get_raw_data("data")


for i in range(5):
    print(f"Example {i}:\n{raw['train'][i]}\n")


train_ds_hf, val_ds_hf, test_ds_hf = get_smaller_datasets(raw)

print(
    f"Dataset lengths: train={len(train_ds_hf)}, val={len(val_ds_hf)}, test={len(test_ds_hf)}"
)

train_loader, val_loader, test_loader, vocab = preprocess_data(
    train_ds_hf, val_ds_hf, test_ds_hf
)
vocab_size = len(vocab)

print(vocab_size, list(vocab.items())[:10])

batch0 = next(iter(train_loader))
print("One batch shapes")
print(
    "x:",
    tuple(batch0.x.shape),
    "lengths:",
    tuple(batch0.lengths.shape),
    "y:",
    tuple(batch0.y.shape),
)
print("Example lengths:", batch0.lengths[:10].tolist())


x_demo = torch.randint(low=0, high=vocab_size, size=(4, 20))
len_demo = torch.tensor([20, 18, 12, 7])
print(
    "LSTM logits shape:", LSTMClassifier(vocab_size=vocab_size)(x_demo, len_demo).shape
)
print(
    "CNN logits shape: ",
    CNNTextClassifier(vocab_size=vocab_size)(x_demo, len_demo).shape,
)

# cnn = CNN(vocab_size, 64)
# cnn.fit(train_loader, val_loader)

# lstm = LSTM(vocab_size, 64)
# lstm.fit(train_loader, val_loader)


set_seed(13)

MAX_EPOCHS = 12
PATIENCE = 3
LR = 1e-3
CLIP = 1.0


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
        clip_grad_norm=CLIP,
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
