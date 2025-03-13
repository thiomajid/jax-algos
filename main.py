import jax
import jax.numpy as jnp


@jax.jit
def selu(x: jax.Array, alpha=1.67, _lambda=1.05):
    return _lambda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


@jax.jit
def square(x: jax.Array):
    return x**2


if __name__ == "__main__":
    key = jax.random.key(12)
    x = jnp.arange(5.0)

    grad_square = jax.vmap(jax.grad(square), in_axes=0)
    print(f"{x}² = {square(x)} and x' = {grad_square(x)}")

    jac_square = jax.jacrev(square)
    print(f"Jacobian of square at {x} is\n {jac_square(x)}")
