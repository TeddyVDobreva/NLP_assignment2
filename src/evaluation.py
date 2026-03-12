
import matplotlib.pyplot as plt

def plot_learning_curves(results, key: str, title: str, ylabel: str):
    plt.figure()
    for res in results:
        hist = res["hist"]
        epochs = [h["epoch"] for h in hist]
        vals = [h[key] for h in hist]
        plt.plot(epochs, vals, label=res["name"])
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(f"plots/learning_curves{res['name']}")
        plt.close()

