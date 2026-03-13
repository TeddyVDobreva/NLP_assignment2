import matplotlib.pyplot as plt
from pathlib import Path
from pandas import DataFrame
import torch

from src.data_handler import _tokenize_data, _get_raw_data, _numericalize

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

LABELS = {0: "World", 1: "Sports", 2: "Business", 3:"Sci/Tech"}

def compare(results):
    rows = []
    for res in results:
        rows.append(
            [
                res["name"],
                res["val"]["acc"],
                res["val"]["f1"],
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
                "train_time_s",
            ],
        )
        .sort_values(by=["val_macro_f1", "val_acc"], ascending=False)
        .reset_index(drop=True)
    )

    print(df_compare)


def plot_learning_curves(results):
    # plot f1
    for res in results:
        hist = res["hist"]
        epochs = [h["epoch"] for h in hist]
        vals = [h["val_f1"] for h in hist]
        plt.plot(epochs, vals, label=res["name"])
    plt.xlabel("epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.savefig("plots/learning_curves__f1")
    plt.close()

    # plot loss
    for res in results:
        hist = res["hist"]
        epochs = [h["epoch"] for h in hist]
        vals = [h["val_loss"] for h in hist]
        plt.plot(epochs, vals, label=res["name"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("plots/learning_curves_loss")
    plt.close()


def plot_confucion_matrix(model, loader, model_name: str, dataset: str) -> None:
    """
    Function evaluates the classification and saves its confusion matrix.

    Arguments:
        y_true: Series- True labels.
        y_predict: Series- Predicted labels.
        model_name: str- Name of the model.
        dataset: str- Dataset identifier.

    Returns: None
    """
    results = model.evaluate(loader)
    y_true = results["y_true"]
    y_predict = results["y_pred"]
    Path("plots").mkdir(exist_ok=True)
    print(classification_report(y_true, y_predict))

    cm = confusion_matrix(y_true, y_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation="vertical")

    plt.savefig(f"plots/confusion_matrix_{model_name}_{dataset}")
    plt.close()


def do_final_evaluation(model, loader, model_name: str, dataset: str):
    # plot confusion matrix
    plot_confucion_matrix(model, loader, model_name, dataset)
    # print results table
    results = model.evaluate(loader)
    rows = []
    rows.append(
        [
            model_name,
            results["acc"],
            results["f1"],
        ]
    )
    print(
        DataFrame(
            rows,
            columns=[
                "model",
                "acc",
                "macro_f1",
            ],
        )
    )
    

def get_misclassified_examples(model, model_name, path, vocab, max_items: int = 8):
    raw_split = _get_raw_data(path)['test']
    model.eval()
    errs = []
    for ex in raw_split:
        tokens = _tokenize_data(ex["text"])
        ids = _numericalize(tokens, vocab)[:64]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        y = int(ex["label"])
        with torch.no_grad():
            logits = model(x, lengths)
            pred = int(logits.argmax(dim=1).item())
        if pred != y:
            snippet = ex["text"].replace("\n", " ")
            snippet = snippet[:250] + ("..." if len(snippet) > 250 else "")
            errs.append((y, pred, snippet))
        if len(errs) >= max_items:
            break
    
    show_errors(model_name, errs)

def show_errors(name: str, errs):
    print(name)
    for i,(y,p,snip) in enumerate(errs):
        print()
        print(f"error {i+1}")
        print("true:", LABELS[y-1], "pred:", LABELS[p-1])
        print("text:", snip)
