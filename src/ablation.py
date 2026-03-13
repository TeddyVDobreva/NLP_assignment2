import time
from typing import Any, Dict, Iterable

from src.evaluation import compare


def ablation(
    model: Any,
    name: str,
    ablation_argument: Dict[str, Iterable[Any]],
    train_loader: Any,
    val_loader: Any,
    **kwargs: Any,
) -> None:
    """
    The function trains a separate model for each hyperparameter value,
    evaluates it on the validation set, and records the training time
    and performance.

    Args:
        model: the model
        name: name of the model
        ablation_argument:contains hyperparameter and its values
        train_loader: data loader containing the training dataset
        val_loader: data loader containing the validation dataset

    Returns:
        None
    """

    hyperparameter_name = list(ablation_argument.keys())[0]
    results: list[dict[str, Any]] = []
    for i, hp in enumerate(ablation_argument[hyperparameter_name]):
        hyperparameter_dic = {hyperparameter_name: hp}
        m = model(**hyperparameter_dic, **kwargs)
        t0 = time.perf_counter()
        hist = m.fit(
            train_loader,
            val_loader,
        )
        total_time = time.perf_counter() - t0
        val = m.evaluate(val_loader)
        results[i] = {
            "name": f"{name} with {hyperparameter_name}={hp}",
            "hist": hist,
            "val": val,
            "time_s_total": total_time,
        }

    compare(results)
