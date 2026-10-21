import sys
import pandas as pd
import matplotlib.pyplot as plt

base = sys.argv[1]
run = 1  # seed 0

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

for row, (model, name) in enumerate([("cnn", "Basic CNN"), ("resnet", "ResNet-18")]):
    res = pd.read_csv(f"{base}/results/{model}/run_{run}/resources.csv")
    em = pd.read_csv(f"{base}/results/{model}/run_{run}/emissions.csv")

    scale = 1 if model == "cnn" else 60
    unit = "s" if model == "cnn" else "min"

    ax = axes[row, 0]
    ax.plot(res["time"] / scale, res["gpu"], label="GPU")
    ax.plot(res["time"] / scale, res["cpu"], label="CPU")
    ax.set_title(f"{name}: utilization")
    ax.set_xlabel(f"Training time ({unit})")
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right" if model == "cnn" else "center right")
    ax.grid(alpha=0.3)

    ax = axes[row, 1]
    ax.plot(em["duration"] / scale, em["gpu_power"], color="tab:green")
    ax.set_title(f"{name}: GPU power")
    ax.set_xlabel(f"Training time ({unit})")
    ax.set_ylabel("Power (W)")
    ax.set_ylim(0, 60)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("hw_utilization_power.png", dpi=200)
