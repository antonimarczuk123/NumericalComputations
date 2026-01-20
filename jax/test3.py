import jax
import jax.numpy as jnp
from jax import grad

key = jax.random.key(0)

def sigmoid(x):
  return 0.5 * (jnp.tanh(x / 2) + 1)

def predict(W, b, inputs):
  return sigmoid(jnp.dot(inputs, W) + b)

inputs = jnp.array([
    [0.52, 1.12,  0.77],
    [0.88, -1.08, 0.15],
    [0.52, 0.06, -1.30],
    [0.74, -2.49, 1.39]
])

targets = jnp.array([1, 1, 0, 1])

def loss(W, b):
  preds = predict(W, b, inputs)
  label_probs = preds * targets + (1 - preds) * (1 - targets)
  return -jnp.sum(jnp.log(label_probs))

Grad_W_loss = grad(loss, 0)
Grad_b_loss = grad(loss, 1)

key, W_key, b_key = jax.random.split(key, 3)
W = jax.random.normal(W_key, (3,))
b = jax.random.normal(b_key, ())

W_grad = Grad_W_loss(W, b)
print(f'{W_grad=}')

b_grad = Grad_b_loss(W, b)
print(f'{b_grad=}')
