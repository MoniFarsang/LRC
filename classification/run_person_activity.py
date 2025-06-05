# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.
# https://github.com/mlech26l/ode-lstms/blob/master/person_activity.py
# Modified by Monika Farsang (2024).

import os
import tensorflow as tf
import argparse
from lstm_cell import LSTMCell
from gru_cell import MGUCell, GRUCell
from lrc_cell import LRC_Cell

from irregular_sampled_datasets import PersonData

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["LRC_sym_elastance", "LRC_asym_elastance", "lstm", "mgu", "gru"], default="LRC_sym_elastance")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--unfolds", default=1, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--id", default=1, type=int)
args = parser.parse_args()

data = PersonData()

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

pixel_input = tf.keras.Input(shape=(data.seq_len, data.feature_size), name="features")
time_input = tf.keras.Input(shape=(data.seq_len, 1), name="time")

rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=True)
dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(data.num_classes))

output_states = rnn((pixel_input, time_input))
y = dense_layer(output_states)

model = tf.keras.Model(inputs=[pixel_input, time_input], outputs=[y])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(args.lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
model.summary()

name = f"{args.model}_{args.size}_unfold_{args.unfolds}_lr_{args.lr}_epochs_{args.epochs}_id_{args.id}"

os.makedirs("learning_curves/person_activity", exist_ok=True)
os.makedirs("ckpt/person_activity", exist_ok=True)

# Fit and evaluate
hist = model.fit(
    x=(data.train_x, data.train_t),
    y=data.train_y,
    batch_size=128,
    epochs=args.epochs,
    validation_data=((data.test_x, data.test_t), data.test_y),
    callbacks=[
        tf.keras.callbacks.CSVLogger(f"learning_curves/person_activity/{name}.csv"),
    ],
)
_, best_test_acc = model.evaluate(
    x=(data.test_x, data.test_t), y=data.test_y, verbose=2
)

os.makedirs("results/person_activity", exist_ok=True)
with open(f"results/person_activity/{name}.csv", "a") as f:
    f.write("{:06f}\n".format(best_test_acc))