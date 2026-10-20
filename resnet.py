import os
import sys
import glob
import ctypes
import time
import threading

# without this tensorflow doesnt find the gpu (libcusolver missing)
for f in glob.glob(sys.prefix + "/lib/python*/site-packages/nvidia/cusolver/lib/libcusolver.so.*"):
    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import psutil
import pynvml
import tensorflow as tf
from codecarbon import EmissionsTracker
import keras
from keras import layers, models
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping

EPOCHS = 30
BATCH_SIZE = 128

# run number comes from run_all.py (python resnet.py 3 -> run 3), default is 1
RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# different seed for every run (weights init, shuffling, augmentation), cnn and resnet use the same seed in the same run
# seeds 0-4 like in the notebook
SEED = RUN - 1
keras.utils.set_random_seed(SEED)  # sets python, numpy and tensorflow seeds
RESULTS = f"results/resnet/run_{RUN}"
os.makedirs(RESULTS, exist_ok=True)
print("GPU:", tf.config.list_physical_devices("GPU"))

# load cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# split train into train (45000) and validation (5000)
# validation is used to pick the best epoch (early stopping), test is only used once at the end
# random split with a fixed seed (42) so its the same split for every run and model
idx = np.random.default_rng(42).permutation(len(x_train))
x_val = x_train[idx[:5000]]
y_val = y_train[idx[:5000]]
x_train = x_train[idx[5000:]]
y_train = y_train[idx[5000:]]

# data augmentation, same for both models. only active during training, not during evaluate
def augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.125, 0.125, fill_mode="reflect"),   # +-4 px on a 32 px image
    ], name="augmentation")


# resnet-18
# made of "residual blocks": 2 conv layers + a shortcut that adds the input back to the output.
# the shortcut lets the gradient flow through deep networks so they can still learn


def residual_block(x, filters, stride):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # if the size changed, the shortcut needs a 1x1 conv so the shapes match for the add
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


inputs = layers.Input((32, 32, 3))

# first conv (original resnet uses 7x7 + maxpool, but that's for 224x224 images,
# for 32x32 cifar images a 3x3 conv is used instead so the image doesnt shrink too fast)
x = augmentation()(inputs)
x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# 4 stages with 2 blocks each = 16 conv layers, + first conv + last dense = 18 layers
x = residual_block(x, 64, 1)
x = residual_block(x, 64, 1)
x = residual_block(x, 128, 2)  # stride 2 halves the image size: 32 -> 16
x = residual_block(x, 128, 1)
x = residual_block(x, 256, 2)  # 16 -> 8
x = residual_block(x, 256, 1)
x = residual_block(x, 512, 2)  # 8 -> 4
x = residual_block(x, 512, 1)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = models.Model(inputs, outputs, name="resnet18")

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# monitor cpu, gpu and ram usage every second in another thread
pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
usage = []
running = True


def monitor():
    psutil.cpu_percent()  # first value is always 0
    start = time.time()
    while running:
        time.sleep(1)
        cpu = psutil.cpu_percent()
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu
        ram = psutil.virtual_memory().percent
        usage.append((round(time.time() - start, 1), cpu, gpu_util, ram))
        tracker.flush()  # makes codecarbon write a row to emissions.csv every second too


# delete old runs so emissions.csv only has this run
if os.path.exists(RESULTS + "/emissions.csv"):
    os.remove(RESULTS + "/emissions.csv")

# only measure energy during training
tracker = EmissionsTracker(project_name="resnet18", output_dir=RESULTS, measure_power_secs=1, log_level="error")
tracker.start()

t = threading.Thread(target=monitor)
t.start()
start = time.time()

callback = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=8,
    restore_best_weights=True
)
history = model.fit(
    x_train, 
    y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(x_val, y_val), 
    verbose=2,
    callbacks=[callback]
)

train_time = time.time() - start
running = False
t.join()
tracker.stop()

# model has the best weights from validation now (restore_best_weights), test them on the test set
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
data = tracker.final_emissions_data

# save usage to csv
with open(RESULTS + "/resources.csv", "w") as f:
    f.write("time,cpu,gpu,ram\n")
    for row in usage:
        f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")

avg_cpu = sum(r[1] for r in usage) / len(usage)
avg_gpu = sum(r[2] for r in usage) / len(usage)
avg_ram = sum(r[3] for r in usage) / len(usage)

print("\n--- results ---")
print("params:", model.count_params())
print("training time:", round(train_time, 2), "s")
print("epochs run: ", len(history.history["val_loss"]))
print("energy:", data.energy_consumed, "kWh")
print("  cpu:", data.cpu_energy, "kWh")
print("  gpu:", data.gpu_energy, "kWh")
print("  ram:", data.ram_energy, "kWh")
print("emissions:", data.emissions, "kg CO2")
print("val accuracy:", round(val_acc, 4))
print("test accuracy:", round(acc, 4))
print("avg cpu %:", round(avg_cpu, 1))
print("avg gpu %:", round(avg_gpu, 1))
print("avg ram %:", round(avg_ram, 1))

# add this run to the summary file shared by all runs
summary = "results/summary.csv"
if not os.path.exists(summary):
    with open(summary, "w") as f:
        f.write("model,run,seed,params,epochs_run,train_time_s,energy_kwh,cpu_energy_kwh,gpu_energy_kwh,ram_energy_kwh,"
                "emissions_kg,val_accuracy,test_accuracy,avg_cpu,avg_gpu,avg_ram\n")
with open(summary, "a") as f:
    f.write(f"resnet,{RUN},{SEED},{model.count_params()},{len(history.history['val_loss'])},{train_time},"
            f"{data.energy_consumed},{data.cpu_energy},{data.gpu_energy},{data.ram_energy},"
            f"{data.emissions},{val_acc},{acc},{avg_cpu},{avg_gpu},{avg_ram}\n")
