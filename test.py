import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jaxopt

import numpy as np

import sklearn.datasets as skd
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from tqdm import trange



class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(200)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(1)(x)
        x = nn.relu(x)
        return x


def mean_absolute_error(model):
    def _apply(variables, X, Y):
        preds = model.apply(variables, X)
        return jnp.mean(jnp.abs(Y - preds.reshape(-1)))
    return _apply


if __name__ == "__main__":
    X, Y = skd.fetch_california_housing(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = Net()
    variables = model.init(jax.random.PRNGKey(42), X_train[:1])
    rng = np.random.default_rng(42)
    solver = jaxopt.OptaxSolver(opt=optax.adam(0.001), fun=mean_absolute_error(model))
    state = solver.init_state(variables, X=X_train[:1], Y=Y_train[:1])
    step = jax.jit(solver.update)
    batch_size = 200

    for _ in (pbar := trange(3000)):
        idx = rng.choice(len(Y_train), batch_size)
        variables, state = step(variables, state, X=X_train[idx], Y=Y_train[idx])
        pbar.set_postfix_str(f"Loss: {state.value:.3f}")

    preds = model.apply(variables, X_test)
    print(f"R2 score: {skm.r2_score(Y_test, preds)}")
    print(f"MAE: {skm.mean_absolute_error(Y_test, preds)}")