from typing import Union

import jax.numpy as jnp
import numpy as np
import jax

from bmi.interface import KeyArray
from bmi.samplers.base import BaseSampler, cast_to_rng
from bmi.benchmark.task import Task


class AdditiveUniformSamplerMulti(BaseSampler):
    def __init__(self, epsilon: Union[float, np.ndarray], dim: int) -> None:
        """
        Represents the distribution P(X, Y) under the model:

        $$X_j \\sim \\mathrm{Uniform}(0, 1)$$
        $$N_j \\sim \\mathrm{Uniform}(-\\epsilon_j, \\epsilon_j)$$
        $$Y_j = X_j + N_j$$

        The MI is computed as the sum of individual MI for each dimension.
        """
        super().__init__(dim_x=dim, dim_y=dim)
        epsilon_array = jnp.asarray(epsilon)

        if epsilon_array.ndim == 0:
            # Broadcast scalar epsilon to all dimensions
            epsilon_array = jnp.broadcast_to(epsilon_array, (dim,))
        elif epsilon_array.shape != (dim,):
            raise ValueError(f"epsilon must be scalar or have length {dim}, got {epsilon_array.shape}")

        if jnp.any(epsilon_array <= 0):
            raise ValueError(f"All elements of epsilon must be positive, got {epsilon_array}")

        self._epsilon = epsilon_array

    def sample(self, n_points: int, rng: Union[int, KeyArray]) -> tuple[np.ndarray, np.ndarray]:
        rng = cast_to_rng(rng)
        x = np.random.uniform(0.0, 1.0, size=(n_points, self.dim_x))
        n = np.random.uniform(-self._epsilon, self._epsilon, size=(n_points, self.dim_x))
        y = x + n
        return x, y

    @staticmethod
    def _mi_per_dimension(epsilon: float) -> float:
        return jnp.where(
            epsilon > 0.5,
            0.25 / epsilon,
            epsilon - jnp.log(2 * epsilon)
        )

    @staticmethod
    def mutual_information_function(epsilon: Union[float, np.ndarray]) -> float:
        eps = jnp.asarray(epsilon)
        if eps.ndim == 0:
            return AdditiveUniformSamplerMulti._mi_per_dimension(eps)
        else:
            mi_per_dim = jax.vmap(AdditiveUniformSamplerMulti._mi_per_dimension)(eps)
            return mi_per_dim.sum()

    def mutual_information(self) -> float:
        return type(self).mutual_information_function(self._epsilon)
    
    def entropy_x(self) -> float:
        """Returns the differential entropy of X in nats."""
        return 0.0  # H(Uniform(0,1)) = log(1) = 0 for all dimensions

    def entropy_y(self) -> float:
        """Returns the differential entropy of Y in nats."""
        def h_per_dim(epsilon_j):
            return jnp.where(
                epsilon_j <= 0.5,
                epsilon_j,
                0.25 / epsilon_j + jnp.log(2 * epsilon_j)
            )
        h_per_dim_vmap = jax.vmap(h_per_dim)(self._epsilon)
        return h_per_dim_vmap.sum()

    def entropy_xy(self) -> float:
        """Returns the joint differential entropy of (X,Y) in nats."""
        # For each dimension, H(X_j, Y_j) = log(2 * epsilon_j)
        return jnp.sum(jnp.log(2 * self._epsilon))
    


def task_multi_additive_noise(
    epsilon: float,
    dim: int,
) -> Task:
    sampler = AdditiveUniformSamplerMulti(epsilon=epsilon, dim=dim)

    return Task(
        sampler=sampler,
        task_id=f"{dim}-dim-additive-{epsilon}",
        task_name=f"Uniform {dim} × {dim} (additive noise={epsilon})",
        task_params={
            "epsilon": epsilon,
            "dim": dim,
        },
    )