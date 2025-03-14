from functools import partial

import jax
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tqdm import tqdm


class LinearRegression:
    def __init__(self, *, rngs: jax.random.PRNGKey, n_features: int, n_outputs: int):
        self.rngs = rngs
        self.n_features = n_features
        self.n_outputs = n_outputs

        self.rngs, w_key = jax.random.split(self.rngs)
        self.w = jax.random.normal(w_key, shape=(self.n_features, self.n_outputs))

        self.rngs, b_key = jax.random.split(self.rngs)
        self.b = jax.random.normal(b_key, shape=(self.n_outputs,))

    @jax.jit
    def __call__(self, x: jax.Array):
        out = x @ self.w + self.b
        return out


def mse_loss(predictions: jax.Array, targets: jax.Array):
    return ((predictions - targets) ** 2).mean()


@jax.jit
@partial(jax.value_and_grad, argnums=(0, 1))
def loss_fn(w, b, x, y):
    predictions = x @ w + b
    return mse_loss(predictions, y)


def train(
    key: jax.random.PRNGKey,
    lr: float,
    num_epochs: int,
    batch_size: int,
    targets: jax.Array,
    labels: jax.Array,
    in_features: int,
    out_features: int,
):
    model = LinearRegression(rngs=key, n_features=in_features, n_outputs=out_features)
    losses = []

    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            # Shuffle the data
            key, subkey = jax.random.split(key)
            indices = jax.random.permutation(subkey, targets.shape[0])

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, targets.shape[0], batch_size):
                batch_indices = indices[i : i + batch_size]
                x_batch = targets[batch_indices]
                y_batch = labels[batch_indices]

                # Compute loss and gradients
                loss, (dw, db) = loss_fn(model.w, model.b, x_batch, y_batch)

                # Update weights and biases
                model.w -= lr * dw
                model.b -= lr * db

                epoch_loss += loss
                num_batches += 1

                # Update progress bar message periodically
                if i % 100 == 0:
                    losses.append(loss)

            # Update progress bar after each epoch
            avg_loss = epoch_loss / num_batches
            pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}"
            )
            pbar.update(1)

    return model, losses


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    n_features = 10
    n_outputs = 3

    # Generate some random data
    num_samples = 1000
    key, w_key = jax.random.split(key)
    true_w = jax.random.normal(w_key, shape=(n_features, n_outputs))

    key, b_key = jax.random.split(key)
    true_b = jax.random.normal(b_key, shape=(n_outputs,))

    key, x_key = jax.random.split(key)
    x_data = jax.random.normal(key, (num_samples, n_features))

    key, noise_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, (num_samples, n_outputs)) * 0.1

    y_data = x_data @ true_w + true_b + noise

    # Train the model
    model, losses = train(
        key=key,
        lr=0.001,
        num_epochs=50,
        batch_size=16,
        targets=x_data,
        labels=y_data,
        in_features=n_features,
        out_features=n_outputs,
    )

    plt.plot(losses)
    plt.xlabel("Train steps")

    plt.ylabel("Loss")
    plt.title(
        "Loss over time:  $MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2$"
    )

    plt.xticks(range(0, len(losses), 10), range(0, len(losses), 10))
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    plt.show()
