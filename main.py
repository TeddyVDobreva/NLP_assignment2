import random
from src.data_handler import get_raw_data, get_smaller_datasets,preprocess_data
import numpy as np
import torch


def set_seed(seed: int = 67) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(13)

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

raw = get_raw_data('data')


for i in range(5):
    print(f"Example {i}:\n{raw['train'][i]}\n")


train_ds_hf, val_ds_hf, test_ds_hf = get_smaller_datasets(raw)

print(
    f"Dataset lengths: train={len(train_ds_hf)}, val={len(val_ds_hf)}, test={len(test_ds_hf)}"
)

train_loader, val_loader, test_loader, vocab = preprocess_data(train_ds_hf, val_ds_hf, test_ds_hf)
vocab_size = len(vocab)

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
