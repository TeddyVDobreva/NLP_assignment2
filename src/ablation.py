import time
from pandas import DataFrame

def ablation(model, ablatioin_argument, train_loader, val_loader, **kwargs):

    hyperparameter_name = list(ablatioin_argument.keys())[0]
    results = [None for _ in ablatioin_argument[hyperparameter_name]]
    for i, hp1 in enumerate(ablatioin_argument[hyperparameter_name]):
        hyperparameter_dic = {hyperparameter_name: hp1}
        m = model(**hyperparameter_dic,**kwargs)
        t0 = time.perf_counter()
        hist = m.fit(
        train_loader,
        val_loader,
        )
        total_time = time.perf_counter() - t0
        val = m.evaluate(val_loader)
        results[i] = {
        "name": model.__class__,
        "hist": hist,
        "val": val,
        "time_s_total": total_time,
        }
        
    
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