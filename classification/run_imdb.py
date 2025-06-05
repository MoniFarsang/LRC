import os
import tensorflow as tf
from lstm_cell import LSTMCell
from gru_cell import MGUCell, GRUCell
from lrc_cell import LRC_Cell
import argparse
import math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["LRC_sym_elastance", "LRC_asym_elastance", "lstm", "mgu", "gru"], default="LRC_sym_elastance")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--unfolds", default=1, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--id", default=1, type=int)
args = parser.parse_args()

class BackupToBestValEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, name):
        super(BackupToBestValEpochCallback, self).__init__()
        self._name = name
        self._best_val_accuracy = 0
        self._train_loss_mse_when_best_val_loss = math.inf

        self._best_epoch = None
        self.copied_weights = None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_accuracy"] >= self._best_val_accuracy:
            self.copied_weights = self.model.get_weights()
            self._best_epoch = epoch
            self._best_val_accuracy = logs["val_accuracy"]
            self._train_loss_mse_when_best_val_loss = logs["loss"]
        if epoch%10 == 0:
            store_weights(self.model, f"ode-lstms/ckpt/imdb/{self._name}_epoch{epoch}.npz")

    def on_train_end(self, logs=None):
        if self.copied_weights is not None:
            print(
                f"Restoring weights to epoch {self._best_epoch} with val_sparse_categorical_accuracy={self._best_val_accuracy:0.4g} (train_loss={self._train_loss_mse_when_best_val_loss:0.4g})"
            )
            self.model.set_weights(self.copied_weights)
        store_weights(self.model, f"ode-lstms/ckpt/imdb/{self._name}_final.npz")
        filename = "ode-lstms/ckpt/summary.txt"
        with open(filename, "a") as f:
            f.write(
                f"Model: {self._name} \nBest epoch: {self._best_epoch}, train loss mse: {self._train_loss_mse_when_best_val_loss:0.4g}, val_accuracy: {self._best_val_accuracy:0.4g})\n\n"

            )

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
    
# Load the IMDB dataset
vocab_size = 20000
maxlen = 256
embed_dim = 64
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x, maxlen=maxlen)

x_val = train_x[:2500]
x_train = train_x[2500:]

y_val = train_y[:2500]
y_train = train_y[2500:]

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

inputs = tf.keras.layers.Input(shape=(maxlen,))
token_emb = tf.keras.layers.Embedding(
    input_dim=vocab_size, output_dim=embed_dim
)
cell_input = token_emb(inputs)

rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)
dense_layer = tf.keras.layers.Dense(1)

output_states = rnn(cell_input)
y = dense_layer(output_states)

model = tf.keras.Model(inputs, y)

model.compile(
    optimizer=tf.keras.optimizers.Adam(args.lr),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
model.summary()

name = f"{args.model}_{args.size}_unfold_{args.unfolds}_lr_{args.lr}_epochs_{args.epochs}_id_{args.id}"

os.makedirs("learning_curves/imdb", exist_ok=True)
os.makedirs("ckpt/imdb", exist_ok=True)
# Fit and evaluate
hist = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=args.epochs,
    validation_data=(x_val, y_val),
    callbacks=[
        tf.keras.callbacks.CSVLogger(f"learning_curves/imdb/{name}.csv"),
        BackupToBestValEpochCallback(name=f"imdb_{name}"),
    ],
)

# Evaluate the model on the test set
_, test_acc = model.evaluate(test_x, test_y, verbose=2)

os.makedirs("results/imdb", exist_ok=True)
with open(f"results/imdb/{name}.csv", "a") as f:
    f.write("{:06f}\n".format(test_acc))