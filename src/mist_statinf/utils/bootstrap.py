# src/mist/utils/bootstrap.py
import numpy as np
from typing import Iterable

def bootstrap_data(arr: np.ndarray, n_resamples: int = 7) -> Iterable[np.ndarray]:
    N = arr.shape[0]
    for _ in range(n_resamples):
        idx = np.random.randint(0, N, size=N)
        yield arr[idx]
