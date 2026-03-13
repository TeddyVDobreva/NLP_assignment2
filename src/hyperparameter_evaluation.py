from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score

from src.models import CNNTextClassifier, LSTMClassifier


def do_hyperparameter_evaluation(
    model: Any,
    hyperparameter1: dict[str, list],
    hyperparameter2: dict[str, list],
    vocab_size: int,
    train_loader: Any,
    validation_loader: Any,
    **kwargs: Any,
) -> None:
    """
    Function iterates over the provided hyperparameters, trains a new model
    for every combination, evaluates it, then stores the result in a 2D
    accuracy matrix. The model skips the unsupported combinations of
    hyperparameters.

    Arguments:
        model: LinearSVC | LogisticRegression- The model which will be
            trained and evaluated.
        hyperparameter1: dict[str, list]- Dictionary containing the 1st
            hyperparameter name.
        hyperparameter2: dict[str, list]- Dictionary containing the 2nd
            hyperparameter name.
        X_train: Any- Training matrix.
        labels_train: pd.Series- Training labels.
        X_validation: Any- Validation matrix.
        labels_validation: pd.Series- Validation labels.
        **kwargs: Any- Additional individual arguments necessary for each
            model.

    Returns: None
    """
    print("Hyper parameter tuning started")
    hp1_name = list(hyperparameter1.keys())[0]
    hp2_name = list(hyperparameter2.keys())[0]

    accuracy_matrix = np.full(
        (len(hyperparameter1[hp1_name]), len(hyperparameter2[hp2_name])), np.nan
    )

    for i, hp1 in enumerate(hyperparameter1[hp1_name]):
        print(f"Current {hp1_name}: {hp1}")

        for j, hp2 in enumerate(hyperparameter2[hp2_name]):
            print(f"Current {hp2_name} {hp2}")
            try:
                hyperparameter_dic = {hp2_name: hp2}

                model_to_tune = model(
                    vocab_size=vocab_size, **hyperparameter_dic, **kwargs
                )

                model_to_tune.fit(
                    train_loader=train_loader,
                    val_loader=validation_loader,
                    lr=hp1,
                )

                dictionary_validation = model_to_tune.evaluate(train_loader)

                validation_accuracy = accuracy_score(
                    dictionary_validation["y_true"], dictionary_validation["y_pred"]
                )

                accuracy_matrix[i, j] = validation_accuracy
            except Exception as e:
                print(f"Skipping combination — {e}")
        print()

    make_heatmap(accuracy_matrix, hyperparameter1, hyperparameter2)


def make_heatmap(
    accuracy_matrix: np.ndarray,
    hyperparameter1: dict[str, list],
    hyperparameter2: dict[str, list],
) -> None:
    """
    Function presents the accuracy matrix by using a heatmap, which shows how
    performance varies for the models when using different hyperparameters.

    Arguments:
        accuracy_matrix: np.ndarray: 2D matrix containing accuracies for
            each hyperparameter combination.
       hyperparameter1: dict[str, list]- Dictionary containing the 1st
            hyperparameter name.
        hyperparameter2: dict[str, list]- Dictionary containing the 2nd
            hyperparameter name.

    Returns: None
    """
    Path("plots").mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    hp1_name = list(hyperparameter1.keys())[0]
    hp2_name = list(hyperparameter2.keys())[0]

    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt=".3f",
        yticklabels=hyperparameter1[hp1_name],
        xticklabels=hyperparameter2[hp2_name],
        cmap="viridis",
    )

    plt.title("Heatmap")
    plt.ylabel(hp1_name)
    plt.xlabel(hp2_name)
    plt.savefig(f"plots/heatmap_{hp1_name}_{hp2_name}")
    plt.close()
