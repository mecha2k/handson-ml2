import numpy as np
import tensorflow as tf
import time

from tensorflow import keras
from tqdm.notebook import trange
from collections import OrderedDict

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


def progress_bar(iteration, total, size=30):
    running = iteration < total
    c = ">" if running else "="
    p = (size - 1) * iteration // total
    fmt = "{{:-{}d}}/{{}} [{{}}]".format(len(str(total)))
    params = [iteration, total, "=" * p + c + "." * (size - p - 1)]
    return fmt.format(*params)


def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(
        ["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])]
    )
    end = "" if iteration < total else "\n"
    print("\r{} - {}".format(progress_bar(iteration, total), metrics), end=end)


def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)


def fashion_mnist():
    (X_train_full, y_train_full), (_, _) = keras.datasets.fashion_mnist.load_data()
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    model = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    n_epochs = 5
    batch_size = 32
    n_steps = len(X_train) // batch_size
    optimizer = keras.optimizers.Nadam(lr=0.01)
    loss_fn = keras.losses.sparse_categorical_crossentropy
    mean_loss = keras.metrics.Mean()
    metrics = [keras.metrics.SparseCategoricalAccuracy()]

    with trange(1, n_epochs + 1, desc="All epochs") as epochs:
        for epoch in epochs:
            with trange(1, n_steps + 1, desc="Epoch {}/{}".format(epoch, n_epochs)) as steps:
                for _ in steps:
                    X_batch, y_batch = random_batch(X_train, y_train)
                    with tf.GradientTape() as tape:
                        y_pred = model(X_batch)
                        main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                        loss = tf.add_n([main_loss] + model.losses)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    for variable in model.variables:
                        if variable.constraint is not None:
                            variable.assign(variable.constraint(variable))
                    status = OrderedDict()
                    mean_loss(loss)
                    status["loss"] = mean_loss.result().numpy()
                    for metric in metrics:
                        metric(y_batch, y_pred)
                        status[metric.name] = metric.result().numpy()
                    steps.set_postfix(status)
                y_pred = model(X_valid)
                status["val_loss"] = np.mean(loss_fn(y_valid, y_pred))
                status["val_accuracy"] = np.mean(
                    keras.metrics.sparse_categorical_accuracy(
                        tf.constant(y_valid, dtype=np.float32), y_pred
                    )
                )
                steps.set_postfix(status)
            for metric in [mean_loss] + metrics:
                metric.reset_states()


def california_housing():
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    input_shape = X_train.shape[1:]
    model = keras.models.Sequential(
        [
            keras.layers.Dense(
                30, activation="selu", kernel_initializer="lecun_normal", input_shape=input_shape
            ),
            keras.layers.Dense(1),
        ]
    )

    model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])
    model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid))

    n_epochs = 5
    batch_size = 32
    n_steps = len(X_train) // batch_size
    optimizer = keras.optimizers.Nadam(lr=0.01)
    loss_fn = keras.losses.mean_squared_error
    mean_loss = keras.metrics.Mean()
    metrics = [keras.metrics.MeanAbsoluteError()]

    for epoch in range(1, n_epochs + 1):
        print("Epoch {}/{}".format(epoch, n_epochs))
        for step in range(1, n_steps + 1):
            X_batch, y_batch = random_batch(X_train_scaled, y_train)
            with tf.GradientTape() as tape:
                y_pred = model(X_batch)
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            for variable in model.variables:
                if variable.constraint is not None:
                    variable.assign(variable.constraint(variable))
            mean_loss(loss)
            for metric in metrics:
                metric(y_batch, y_pred)
            print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
        print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
        for metric in [mean_loss] + metrics:
            metric.reset_states()


def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    keras.backend.clear_session()

    # fashion_mnist()
    california_housing()


if __name__ == "__main__":
    main()
