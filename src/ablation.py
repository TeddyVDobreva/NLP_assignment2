import time
from src.evaluation import compare


def ablation(model, name, ablatioin_argument, train_loader, val_loader, **kwargs):

    hyperparameter_name = list(ablatioin_argument.keys())[0]
    results = [None for _ in ablatioin_argument[hyperparameter_name]]
    for i, hp in enumerate(ablatioin_argument[hyperparameter_name]):
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
