import numpy as np
from typing import Dict, List, Tuple, Optional, TypedDict
from numpy.typing import NDArray

# BMI Tasks
import bmi.benchmark.tasks.bivariate_normal as binormal
import bmi.benchmark.tasks.additive_noise as additive_noise
import bmi.benchmark.tasks.bimodal_gaussians as bimodal_gaussians
from bmi.benchmark.tasks.student import (
    task_student_identity,
    task_student_dense,
    task_student_sparse,
    task_student_2pair,
)
from bmi.benchmark.tasks.multinormal import (
    task_multinormal_dense,
    task_multinormal_sparse,
    task_multinormal_2pair,
    task_multinormal_lvm,
)
from .multiadditive_noise import task_multi_additive_noise

# Transformations
from bmi.benchmark.tasks.asinh import transform_asinh_task as asinh
from bmi.benchmark.tasks.half_cube import transform_half_cube_task as half_cube
from bmi.benchmark.tasks.normal_cdf import transform_normal_cdf_task as normal_cdfise
from bmi.benchmark.tasks.wiggly import transform_wiggly_task as wigglify


# Task function mappings
STUDENT_TASKS = {
    "dense": task_student_dense,
    "sparse": task_student_sparse,
    "2pair": task_student_2pair,
}

MULTINORMAL_TASKS = {
    "dense": task_multinormal_dense,
    "sparse": task_multinormal_sparse,
    "2pair": task_multinormal_2pair,
    "lvm": task_multinormal_lvm,
}


class DatasetResult(TypedDict):
    dataset: NDArray[np.float32]
    parameters: Dict[str, object]


def sample_hyperparameters(
    task_type: str,
    max_dim: int = 5,
    corr_range: Tuple[float, float] = (-0.9, 0.9),
    epsilon_range: Tuple[float, float] = (0.1, 2.0),
    dof_range: Tuple[int, int] = (1, 10),
    off_diag_range: Tuple[float, float] = (0.0, 0.5),
    n_interacting_choices: Tuple[int, ...] = (1,),
    strength_range: Tuple[float, float] = (0.1, 5.0),
    alpha_range: Tuple[float, float] = (0.0, 1.0),
    lambda_range: Tuple[float, float] = (0.1, 3.0),
    beta_range: Tuple[float, float] = (0.0, 1.5),
    eta_range: Tuple[float, float] = (0.1, 5.0),
) -> Dict[str, float]:
    """
    Sample hyperparameters for a given task type.

    Args:
        task_type: Type of task (e.g., 'binormal', 'dense', etc.)
        max_dim: Maximum dimension for tasks involving multiple variables
        ... (other ranges for sampling)

    Returns:
        Dictionary of sampled parameters.
    """
    param_ranges = {
        "corr": corr_range,
        "epsilon": epsilon_range,
        "df": dof_range,
        "off_diag": off_diag_range,
        "n_interacting": (n_interacting_choices[0], max_dim),
        "strength": strength_range,
        "alpha": alpha_range,
        "lambd": lambda_range,
        "beta": beta_range,
        "eta": eta_range,
    }

    param_specs = {
        "identity": ["df"],
        "binormal": ["corr"],
        "additive_noise": ["epsilon"],
        "multi_additive_noise": ["epsilon"],
        "bimodal_gaussians": ["corr"],
        "dense": ["off_diag"],
        "sparse": ["n_interacting", "strength"],
        "lvm": ["n_interacting", "alpha", "lambd", "beta", "eta"],
        "2pair": ["strength"],
    }

    if task_type not in param_specs:
        raise ValueError(f"Unknown task type: {task_type}")

    sampled_params = {}
    for param in param_specs[task_type]:
        if param not in param_ranges:
            raise ValueError(f"No range defined for parameter '{param}'.")

        low, high = param_ranges[param]

        if param == "n_interacting":
            sampled_params[param] = np.random.randint(int(low), int(high) + 1)
        else:
            sampled_params[param] = np.random.uniform(low, high)

    return sampled_params


def generate_mi_dataset(
    base_distribution: str,
    transforms: List[str],
    n_row: int,
    x_dim: int = 1,
    y_dim: int = 1,
    multi_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> DatasetResult:
    """
    Generate a synthetic dataset for mutual information estimation.

    Args:
        base_distribution: Base distribution name (e.g., 'binormal', 'multi_normal')
        transforms: List of transformations to apply (e.g., ['normal_cdfise', 'wigglify'])
        n_row: Number of samples to generate
        x_dim: Dimension of X variable
        y_dim: Dimension of Y variable
        multi_type: Subtype for multivariate tasks (e.g., 'dense', 'sparse')
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - 'dataset': Array of shape (n_row, x_dim + y_dim)
            - 'parameters': Metadata including MI, transforms, etc.
    """
    # Set sampling seed
    sample_seed = seed if seed is not None else np.random.randint(0, 10_000)

    # Validate dimensions and multi_type
    if base_distribution in ("multinormal", "student") and x_dim > 1:
        if multi_type not in MULTINORMAL_TASKS and multi_type not in STUDENT_TASKS:
            raise ValueError(f"Invalid or missing multi_type for {base_distribution}: {multi_type}")

    # Sample hyperparameters and create base task
    task = _create_base_task(base_distribution, x_dim, y_dim, multi_type, sample_seed)

    # Apply transformations
    task = _apply_transforms(task, transforms)

    # Sample data
    X, Y = task.sample(n_row, seed=sample_seed)
    dataset = np.hstack((X, Y)).astype(np.float32)

    # Compute mutual information
    mutual_info = float(task.sampler.mutual_information())

    # Build result
    result: DatasetResult = {
        "dataset": dataset,
        "parameters": {
            "base_distribution": base_distribution,
            "transforms": transforms.copy(),
            "mutual_information": mutual_info,
            "sample_seed": int(sample_seed),
        }
    }

    if x_dim > 1:
        result["parameters"]["x-dim and y-dim"] = (x_dim, y_dim)
        result["parameters"]["multi-type"] = multi_type

    return result


def _create_base_task(
    base_distribution: str,
    x_dim: int,
    y_dim: int,
    multi_type: Optional[str],
    seed: int,
) -> object:
    """Create the base task based on distribution and parameters."""
    np.random.seed(seed)  # Ensure consistent sampling within task creation

    if base_distribution == "binormal":
        corr = sample_hyperparameters("binormal")["corr"]
        return binormal.task_bivariate_normal(gaussian_correlation=corr)

    if base_distribution == "additive_noise":
        epsilon = sample_hyperparameters("additive_noise")["epsilon"]
        return additive_noise.task_additive_noise(epsilon=epsilon)

    if base_distribution == "multi_additive_noise":
        if x_dim != y_dim:
            raise ValueError("For multi_additive_noise, x_dim must equal y_dim.")
        epsilon = [sample_hyperparameters("multi_additive_noise")["epsilon"] for _ in range(x_dim)]
        return task_multi_additive_noise(epsilon=epsilon, dim=x_dim)

    if base_distribution == "bimodal_gaussians":
        corr = sample_hyperparameters("bimodal_gaussians")["corr"]
        return bimodal_gaussians.task_bimodal_gaussians(gaussian_correlation=corr)

    if base_distribution in ("student", "multi_student", "bistudent"):
        df = sample_hyperparameters("identity")["df"]
        if x_dim == 1:
            return task_student_identity(dim_x=x_dim, dim_y=y_dim, df=df)
        if multi_type not in STUDENT_TASKS:
            raise ValueError(f"Unknown student task type: {multi_type}")
        dist_args = sample_hyperparameters(multi_type, max_dim=x_dim)
        return STUDENT_TASKS[multi_type](dim_x=x_dim, dim_y=y_dim, df=df, **dist_args)

    if base_distribution == "multi_normal":
        if multi_type not in MULTINORMAL_TASKS:
            raise ValueError(f"Unknown multinormal task type: {multi_type}")
        dist_args = sample_hyperparameters(multi_type, max_dim=x_dim)
        return MULTINORMAL_TASKS[multi_type](dim_x=x_dim, dim_y=y_dim, **dist_args)

    raise ValueError(f"Unknown base_distribution: {base_distribution}")


def _apply_transforms(task: object, transforms: List[str]) -> object:
    """Apply a sequence of transformations to the task."""
    transform_map = {
        "normal_cdfise": normal_cdfise,
        "wigglify": wigglify,
        "halfcube": half_cube,
        "asinh": asinh,
        "base": lambda t: t, 
    }

    for transform_name in transforms:
        transform_fn = transform_map.get(transform_name)
        if transform_fn is None:
            raise ValueError(f"Unknown transform: {transform_name}")
        task = transform_fn(task)

    return task


if __name__ == "__main__":
    result = generate_mi_dataset("binormal", ["base"], n_row=12)
    print("Dataset shape:", result["dataset"].shape)
    print("Mutual Information:", result["parameters"]["mutual_information"])