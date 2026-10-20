import os
import sys
import csv
import time
import shutil
import statistics
import subprocess

RUNS = 5
PAUSE = 30  # seconds to let the gpu cool down between runs so one run doesnt affect the next

# start fresh so old results dont get mixed in
if os.path.exists("results"):
    shutil.rmtree("results")
os.makedirs("results")

# alternate cnn and resnet so both get similar conditions (temperature, background stuff, etc)
# and swap which one goes first every run (cnn first on runs 1,3,5, resnet first on 2,4)
for run in range(1, RUNS + 1):
    order = ["cnn.py", "resnet.py"] if run % 2 == 1 else ["resnet.py", "cnn.py"]
    for script in order:
        print(f"\n########## {script} run {run}/{RUNS} ##########", flush=True)
        # each run is a new python process so tensorflow starts clean every time
        subprocess.run([sys.executable, script, str(run)], check=True)
        time.sleep(PAUSE)

# read all runs from the summary
with open("results/summary.csv") as f:
    rows = list(csv.DictReader(f))

columns = ["params", "epochs_run", "train_time_s", "energy_kwh", "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh",
           "emissions_kg", "val_accuracy", "test_accuracy", "avg_cpu", "avg_gpu", "avg_ram"]

# mean and standard deviation of every column for each model
with open("results/means.csv", "w") as f:
    f.write("model,runs," + ",".join(c + "_mean," + c + "_std" for c in columns) + "\n")
    for model in ["cnn", "resnet"]:
        model_rows = [r for r in rows if r["model"] == model]
        line = [model, str(len(model_rows))]
        print(f"\n--- {model} (mean of {len(model_rows)} runs) ---")
        for c in columns:
            values = [float(r[c]) for r in model_rows]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0
            line += [str(mean), str(std)]
            print(f"{c}: {mean:.6g} (+- {std:.3g})")
        f.write(",".join(line) + "\n")

print("\nsaved results/summary.csv (every run) and results/means.csv (mean per model)")
