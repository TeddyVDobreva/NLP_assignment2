import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

MAX_EPOCHS = 50
PATIENCE = 3
LR = 0.001


def _evaluation_loop(model, loader) -> dict:
    model.eval()
    all_y = []
    all_pred = []
    total_loss = 0.0
    n = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            x = batch.x
            lengths = batch.lengths
            y = batch.y

            logits = model(x, lengths)
            loss = loss_fn(logits, y)

            pred = logits.argmax(dim=1)
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            total_loss += loss.item() * y.size(0)
            n += y.size(0)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    return {
        "loss": total_loss / max(1, n),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _training_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    lr: float = LR,
    max_epochs: int = MAX_EPOCHS,
    weight_decay: float = 0.0,
    patience: int | None = PATIENCE,
) -> list:
    """
    Train the model, optionally with early stopping on validation loss.

    If clip_grad_norm is not None, gradients are clipped by global norm after backward.
    We log the pre clipping total gradient norm each epoch.
    """
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    hist = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.perf_counter()

        total_loss = 0.0
        n = 0
        correct = 0

        grad_norms = []

        for batch in train_loader:
            x = batch.x
            lengths = batch.lengths
            y = batch.y

            optim.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()

            # Measure global grad norm before clipping.
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.data.norm(2).item()
                total_norm_sq += param_norm * param_norm
            total_norm = float(total_norm_sq**0.5)
            grad_norms.append(total_norm)

            optim.step()

            total_loss += loss.item() * y.size(0)
            n += y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

        train_loss = total_loss / max(1, n)
        train_acc = correct / max(1, n)
        val = _evaluation_loop(model, val_loader)
        dt = time.perf_counter() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_f1": val["f1"],
            "time_s": dt,
            "grad_norm_mean": float(np.mean(grad_norms))
            if len(grad_norms)
            else float("nan"),
            "grad_norm_p95": float(np.percentile(grad_norms, 95))
            if len(grad_norms)
            else float("nan"),
            "grad_norm_max": float(np.max(grad_norms))
            if len(grad_norms)
            else float("nan"),
        }
        hist.append(record)

        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['acc']:.4f} f1 {val['f1']:.4f} | "
            f"grad norm mean {record['grad_norm_mean']:.2f} max {record['grad_norm_max']:.2f} | "
            f"time {dt:.1f}s"
        )

        if patience is not None:
            if val["loss"] < best_val - 1e-6:
                best_val = val["loss"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print("Early stopping triggered, restoring best parameters.")
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break

    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)

    return hist


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        num_classes: int = 4,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=bidirectional,
        )
        rep_dim = hidden_dim * (2 if bidirectional else 1)
        self.pool = nn.AvgPool1d(kernel_size=3, stride=1)
        self.rep_dropout = nn.Dropout(dropout)
        self.res = nn.Linear(rep_dim - 2, num_classes)
        self.fc = nn.Softmax(0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.emb_dropout(self.embedding(x))  # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)  # h_n: (num_layers * dirs, B, H)
        h_last = h_n[-1]  # last layer, last direction
        rep = self.pool(h_last)
        rep = self.rep_dropout(rep)
        res = self.res(rep)
        return self.fc(res)

    def evaluate(self, loader):
        return _evaluation_loop(self, loader)

    def fit(
        self,
        train_loader,
        val_loader,
        lr: float = LR,
        max_epochs: int = MAX_EPOCHS,
        weight_decay: float = 0.0,
        patience: int | None = PATIENCE,
    ):
        _training_loop(
            self, train_loader, val_loader, lr, max_epochs, weight_decay, patience
        )


class CNNTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple = (3, 4),
        dropout: float = 0.3,
        pad_idx: int = 0,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim, out_channels=num_filters, kernel_size=k
                )
                for k in kernel_sizes
            ]
        )
        self.pool = nn.AvgPool1d(kernel_size=3, stride=1)
        self.rep_dropout = nn.Dropout(dropout)
        self.res = nn.Linear((num_filters * len(kernel_sizes) - 2), num_classes)
        self.fc = nn.Softmax(0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        emb = self.emb_dropout(self.embedding(x))  # (B, T, E)
        emb_t = emb.transpose(1, 2)  # (B, E, T) for Conv1d
        pooled = []
        for conv in self.convs:
            z = torch.relu(conv(emb_t))  # (B, F, T-k+1)
            p = torch.max(z, dim=2).values  # (B, F)
            pooled.append(p)
        rep = torch.cat(pooled, dim=1)  # (B, F * |K|)
        rep = self.pool(rep)
        rep = self.rep_dropout(rep)
        res = self.res(rep)
        return self.fc(res)

    def evaluate(self, loader):
        return _evaluation_loop(self, loader)

    def fit(
        self,
        train_loader,
        val_loader,
        lr: float = LR,
        max_epochs: int = MAX_EPOCHS,
        weight_decay: float = 0.0,
        patience: int | None = PATIENCE,
    ):
        _training_loop(
            self, train_loader, val_loader, lr, max_epochs, weight_decay, patience
        )
