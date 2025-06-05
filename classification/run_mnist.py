# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.
# https://github.com/mlech26l/ode-lstms/blob/master/et_smnist.py
# Modified by Monika Farsang (2024).

import os
import tensorflow as tf
from lstm_cell import LSTMCell
from gru_cell import MGUCell, GRUCell
from lrc_cell import LRC_Cell
import argparse
import math
import numpy as np

seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["LRC_sym_elastance", "LRC_asym_elastance", "lstm", "mgu", "gru"], default="LRC_sym_elastance")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--unfolds", default=1, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--id", default=1, type=int)
parser.add_argument("--verbose", default=1, type=int)
args = parser.parse_args()

def store_weights(model, filename):
    """
    Stores weights of model in a Numpy file. This function is needed instead of the built-in weight saving
    methods because layer names may be different with with TimeDistributed and non-TimeDistributed version of the model
    """
    serial = {}
    for v in model.variables:
        name = v.name
        # Remove "rnn/" from start
        if name.startswith("rnn/"):
            name = name[len("rnn/") :]
        if name in serial.keys():
            raise ValueError(f"Duplicate weight name: {name}")
        serial[name] = v.numpy()
    np.savez(filename, **serial)
    
class BackupToBestValEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, name):
        super(BackupToBestValEpochCallback, self).__init__()
        self._name = name
        self._best_val_accuracy = 0
        self._train_loss_mse_when_best_val_loss = math.inf

        self._best_epoch = None
        self.copied_weights = None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_sparse_categorical_accuracy"] >= self._best_val_accuracy:
            self.copied_weights = self.model.get_weights()
            self._best_epoch = epoch
            self._best_val_accuracy = logs["val_sparse_categorical_accuracy"]
            self._train_loss_mse_when_best_val_loss = logs["loss"]
        if (epoch+1)%10 == 0:
            store_weights(self.model, f"ode-lstms/ckpt/psmnist/{self._name}_epoch{epoch}.npz")

    def on_train_end(self, logs=None):
        if self.copied_weights is not None:
            print(
                f"Restoring weights to epoch {self._best_epoch} with val_sparse_categorical_accuracy={self._best_val_accuracy:0.4g} (train_loss={self._train_loss_mse_when_best_val_loss:0.4g})"
            )
            self.model.set_weights(self.copied_weights)
        store_weights(self.model, f"ode-lstms/ckpt/psmnist/{self._name}_final.npz")
        filename = "ode-lstms/ckpt/summary.txt"
        with open(filename, "a") as f:
            f.write(
                f"Model: {self._name} \nBest epoch: {self._best_epoch}, train loss mse: {self._train_loss_mse_when_best_val_loss:0.4g}, val_accuracy: {self._best_val_accuracy:0.4g})\n\n"
            )

# Load MNIST dataset
(train_images, train_labels), (test_images,test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255
test_images = test_images / 255
train_images = train_images.reshape((train_images.shape[0], -1, 1))
test_images = test_images.reshape((test_images.shape[0], -1, 1))
perm = rng.permutation(train_images.shape[1])
train_images = train_images[:, perm]
test_images = test_images[:, perm]

X_train = train_images[:50000]
X_valid = train_images[50000:]
X_test = test_images
Y_train = train_labels[:50000]
Y_valid = train_labels[50000:]
Y_test = test_labels

if args.model == "lstm":
    cell = LSTMCell(units=args.size)
elif args.model == "mgu":
    cell = MGUCell(units=args.size)
elif args.model == "gru":
    cell = GRUCell(units=args.size)
elif args.model == "LRC_sym_elastance":
    cell = LRC_Cell(units=args.size, ode_unfolds=args.unfolds, elastance_type="symmetric")
elif args.model == "LRC_asym_elastance":
    cell = LRC_Cell(units=args.size, ode_unfolds=args.unfolds, elastance_type="asymmetric")
else:
    raise ValueError("Unknown model type '{}'".format(args.model))

inputs = tf.keras.Input(shape=(X_train.shape[1], 1), name="pixel")

rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)
dense_layer = tf.keras.layers.Dense(10)

output_states = rnn(inputs)
y = dense_layer(output_states)

model = tf.keras.Model(inputs=inputs, outputs=y)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(args.lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
model.summary()

name = f"{args.model}_{args.size}_unfold_{args.unfolds}_lr_{args.lr}_epochs_{args.epochs}_id_{args.id}"

os.makedirs("learning_curves/psmnist", exist_ok=True)
os.makedirs("ckpt/psmnist", exist_ok=True)

# Fit and evaluate
hist = model.fit(
    x=X_train,
    y=Y_train,
    batch_size=64,
    epochs=args.epochs,
    validation_data=(X_valid, Y_valid),
    verbose = args.verbose,
    callbacks=[
        tf.keras.callbacks.CSVLogger(f"learning_curves/psmnist/{name}.csv"),
        BackupToBestValEpochCallback(name=name)
    ])

_, best_test_acc = model.evaluate(
    x=X_test, y=Y_test, verbose=2
)

os.makedirs("results/psmnist", exist_ok=True)
with open(f"results/psmnist/{name}.csv", "a") as f:
    f.write("{:06f}\n".format(best_test_acc))