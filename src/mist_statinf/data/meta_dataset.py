import copy
import os
import json
import random
import numpy as np
import pickle
from tqdm import tqdm
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional, List, Any

from mist_statinf.data.distribution_generator import generate_mi_dataset

def ensure_directory_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


class MIRegressionGenerator:
    def __init__(
        self,
        number_meta_datapoints: int,
        n_row_range: Tuple[int, int],
        meta_distribution: Dict[str, Any],
        n_dim: str, 
        output_folder: str
    ):
        """
        Mutual Information regression dataset generator.

        Args:
            number_meta_datapoints: Number of datasets to generate
            n_row_range: Range for the number of rows in each dataset
            meta_distribution: Dictionary defining parameter ranges for distributions
            n_dim: Dimension range as string "min-max" (e.g., "2-32")
            output_folder: Path to folder where datasets and metadata will be stored
        """
        self.number_meta_datapoints = number_meta_datapoints
        self.n_row_range = n_row_range
        self.meta_distribution = meta_distribution
        self.output_folder = output_folder
        self.n_dim = n_dim

        ensure_directory_exists(output_folder)

    def generate(self, grid_params: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Generates the meta-dataset, either by sampling row counts randomly or using a predefined grid.

        Args:
            grid_params: Dictionary containing grid parameters with keys 'start', 'stop', and 'num'

        Returns:
            Meta information about the generated dataset
        """
        meta_dataset = []
        meta_info = {
            "number_meta_datapoints": self.number_meta_datapoints,
            "n_row_range": self.n_row_range,
            "grid_params": grid_params.copy() if grid_params else None,
            "meta_distribution": self.meta_distribution,
            "n_dim": self.n_dim,
            "datasets": []
        }

        # Determine row values to use (either sampled or from a grid)
        if grid_params:
            n_row_values = np.linspace(
                start=grid_params['start'],
                stop=grid_params['stop'],
                num=grid_params['num'],
                dtype=int
            )
        else:
            n_row_values = [
                random.randint(self.n_row_range[0], self.n_row_range[1])
                for _ in range(self.number_meta_datapoints)
            ]

        idx = 0
        for n_row in tqdm(n_row_values, desc="Generating meta-dataset"):
            if grid_params:
                # For grid sampling, iterate through all distributions
                for distribution_choice in self.meta_distribution:
                    dim = self._get_random_dimension(distribution_choice)
                    for _ in range(self.number_meta_datapoints):
                        meta_dataset, meta_info, idx = self._generate_meta_datapoint(
                            n_row, distribution_choice, meta_dataset, meta_info, idx, dim
                        )
            else:
                # For random sampling, choose one distribution per n_row
                distribution_choice = random.choice(list(self.meta_distribution))
                dim = self._get_random_dimension(distribution_choice)
                meta_dataset, meta_info, idx = self._generate_meta_datapoint(
                    n_row, distribution_choice, meta_dataset, meta_info, idx, dim
                )

        self._save_meta_data(meta_dataset, meta_info)
        return meta_info

    def _get_random_dimension(self, distribution_choice: str) -> int:
        """Get random dimension for multivariate distributions."""
        if distribution_choice.startswith('multi'):
            dim_min, dim_max = map(int, self.n_dim.split("-"))
            return random.randint(dim_min, dim_max)
        return 1  # Default dimension

    def _generate_meta_datapoint(
        self,
        n_row: int,
        distribution_choice: str,
        meta_dataset: List[Dict[str, Any]],
        meta_info: Dict[str, Any],
        idx: int,
        dim: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
        """
        Generates a single meta-datapoint and updates dataset lists.

        Args:
            n_row: Number of rows in the dataset
            distribution_choice: Chosen distribution
            meta_dataset: List to store meta-data points
            meta_info: Dictionary storing meta information
            idx: Current index
            dim: Dimension for multivariate distributions

        Returns:
            Updated meta_dataset, meta_info, and index
        """
        distribution_params = distribution_choice.split('-')
        base_distribution = distribution_params[0]
        transforms = distribution_params[1:]
        
        multi_type = None
        x_dim = y_dim = dim

        # Handle multivariate distribution types
        if base_distribution.startswith('multi_'):
            distribution_parts = base_distribution.split('_')
            if len(distribution_parts) > 1 and distribution_parts[1] in ['normal', 'student']:
                multi_type = transforms[0] if transforms else None
                transforms = transforms[1:] if transforms else []

        # Generate the dataset
        dataset = generate_mi_dataset(
            n_row=n_row,
            base_distribution=base_distribution,
            transforms=transforms,
            x_dim=x_dim,
            y_dim=y_dim,
            multi_type=multi_type
        )

        target = np.array([dataset['parameters']['mutual_information']])

        # Create metadata entries
        meta_datapoint_info = {
            "id": idx,
            "n_row": n_row,
            "distribution": distribution_choice,
            "dimension": dim,
            "target": float(target[0])
        }
        
        meta_info["datasets"].append(meta_datapoint_info)

        meta_datapoint = copy.deepcopy(meta_datapoint_info)
        meta_datapoint["source"] = dataset['dataset']
        meta_datapoint["target"] = target

        meta_dataset.append(meta_datapoint)

        return meta_dataset, meta_info, idx + 1

    def _save_meta_data(self, meta_dataset: List[Dict[str, Any]], meta_info: Dict[str, Any]):
        """Saves meta dataset and meta information to files."""
        dataset_file = os.path.join(self.output_folder, "meta_dataset.pkl")
        with open(dataset_file, "wb") as f:
            pickle.dump(meta_dataset, f)

        def sanitize(obj):
            # Handle OmegaConf DictConfig
            if isinstance(obj, DictConfig):
                return {k: sanitize(v) for k, v in obj.items()}
            # Handle dict
            elif isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            # Handle list/tuple
            elif isinstance(obj, (list, tuple)):
                return [sanitize(item) for item in obj]
            # Handle NumPy types
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return sanitize(obj.tolist())
            elif isinstance(obj, np.bool_):
                return bool(obj)
            # Handle other non-serializable types
            elif isinstance(obj, (set, frozenset)):
                return list(obj)
            # Pass through basic types
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            # Fallback: convert to string (or omit)
            else:
                return str(obj)  # or f"<{type(obj).__name__}>"

        meta_dict = sanitize(meta_info)
        meta_info_file = os.path.join(self.output_folder, "meta_info.json")
        with open(meta_info_file, "w") as f:
            json.dump(meta_dict, f, indent=4)