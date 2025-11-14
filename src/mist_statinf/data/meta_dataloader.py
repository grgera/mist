import os
import pickle
import torch
import glob
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List, Any
import pytorch_lightning as pl
import torch.nn.functional as F

def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    padding_value: int = 0,
    target_feature_dim: Optional[int] = 64
) -> Dict[str, torch.Tensor]:
    """
    Collate function for padding variable-length sequences.

    Args:
        batch: List of data samples
        padding_value: Value to use for padding
        target_feature_dim: Target feature dimension. If None, uses max dimension in batch

    Returns:
        Dictionary containing padded tensors and additional information
    """
    # Separate sources and targets
    sources = [item["source"] for item in batch]
    targets = [item["target"] for item in batch]

    # Collect other metadata
    other_entries = {
        key: [item[key] for item in batch]
        for key in batch[0]
        if key not in ["source", "target"]
    }

    # Determine target feature dimension
    feature_dims = [x.shape[1] for x in sources]
    if target_feature_dim is None:
        target_feature_dim = max(feature_dims)

    # Resize features to target dimension
    def resize_features(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = x.shape[1]
        if current_dim < target_dim:
            # Pad features
            pad_width = target_dim - current_dim
            return F.pad(x, (0, pad_width), value=padding_value)
        elif current_dim > target_dim:
            # Truncate features
            return x[:, :target_dim]
        return x

    resized_sources = [resize_features(x, target_feature_dim) for x in sources]

    # Pad sequences along sequence length dimension
    lengths = torch.tensor([x.shape[0] for x in resized_sources])
    padded_sources = pad_sequence(
        resized_sources,
        batch_first=True,
        padding_value=padding_value
    )  # (B, max_N, D)

    # Create padding mask
    batch_size, max_seq_len, _ = padded_sources.shape
    device = padded_sources.device
    padding_mask = torch.arange(max_seq_len, device=device).expand(batch_size, max_seq_len) >= lengths.unsqueeze(1)

    # Stack targets
    stacked_targets = torch.stack(targets, dim=0).squeeze(1)

    return {
        "source": padded_sources,
        "padding_mask": padding_mask,
        "target": stacked_targets,
        "input_lengths": lengths,
        **other_entries
    }


class MetaStatDataset(Dataset):
    """Dataset for mutual information regression tasks."""

    def __init__(self, data: List[Dict[str, Any]]):
        """
        Custom dataset for mutual information regression.

        Args:
            data: List of dictionaries containing dataset samples
        """
        self.data = data
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate data format and required keys."""
        if not self.data:
            raise ValueError("Data list cannot be empty")

        required_keys = {"source", "target", "id"}
        for i, item in enumerate(self.data):
            if not all(key in item for key in required_keys):
                raise ValueError(f"Item {i} missing required keys. Expected: {required_keys}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get data sample by index.

        Args:
            idx: Index of sample to retrieve

        Returns:
            Sample dictionary with tensor conversions
        """
        sample = self.data[idx].copy()

        # Convert to tensors
        sample["source"] = torch.as_tensor(sample["source"], dtype=torch.float32)
        sample["target"] = torch.as_tensor(sample["target"], dtype=torch.float32)

        return sample


class MetaStatDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for mutual information regression."""

    def __init__(
        self,
        train_folder: str,
        val_folder: str,
        test_folder: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        target_feature_dim: Optional[int] = 64,
        filter_dim: Optional[int] = None
    ):
        """
        Initialize data module.

        Args:
            train_folder: Path to training data folder
            val_folder: Path to validation data folder
            test_folder: Path to test data folder (optional)
            batch_size: Batch size for all dataloaders
            num_workers: Number of workers for data loading
            target_feature_dim: Target feature dimension for padding
            filter_dim: Target feature dimension to filter
        """
        super().__init__()
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.test_folder = test_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_feature_dim = target_feature_dim
        self.filter_dim = filter_dim

        self.train_data: Optional[MetaStatDataset] = None
        self.val_data: Optional[MetaStatDataset] = None
        self.test_data: Optional[MetaStatDataset] = None


    def _load_dataset(self, folder_path: str) -> MetaStatDataset:
        """Load dataset from folder. Supports single file or multiple parts."""

        pattern = os.path.join(folder_path, "meta_dataset_part*.pkl")
        part_files = sorted(glob.glob(pattern))

        # If no part files, fall back to original single file
        if not part_files:
            data_path = os.path.join(folder_path, "meta_dataset.pkl")
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"No meta_dataset.pkl or meta_dataset_part*.pkl found in {folder_path}"
                )
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        else:
            # Load and combine all parts
            data = []
            for part_file in part_files:
                with open(part_file, "rb") as f:
                    part_data = pickle.load(f)
                    if isinstance(part_data, list):
                        data.extend(part_data)
                    else:
                        raise TypeError(
                            f"Expected list in {part_file}, but got {type(part_data)}"
                        )
            print(f"Loaded combined dataset from {len(part_files)} part(s): total {len(data)} samples")

        # Infer the dimension from the shape of the 'source' array if the
        # 'dimension' key is missing. Assumes source.shape[1] is 2 * dimension.
        patched_count = 0
        errors_found = 0
        for point in data:
            if 'dimension' not in point:
                try:
                    # Infer dimension: d = features / 2
                    feature_dim = point['source'].shape[1]
                    if feature_dim % 2 != 0:
                        print(f"Warning: Skipping data point with odd feature dimension {feature_dim}. Cannot infer dimension.")
                        errors_found += 1
                        continue # Skip this malformed point

                    point['dimension'] = feature_dim // 2
                    patched_count += 1
                except IndexError:
                    print(f"Warning: Skipping data point with malformed source shape {point['source'].shape}.")
                    errors_found += 1
                    continue # Skip this malformed point

        if patched_count > 0:
            print(f"Patched {patched_count} data points by inferring dimension from source shape.")
        if errors_found > 0:
            print(f"Found and skipped {errors_found} malformed data points.")

        if self.filter_dim is not None:
            print(f"Filtering dataset to keep only dimension: {self.filter_dim}")
            original_len = len(data)
            data = [p for p in data if p.get('dimension') == self.filter_dim]
            print(f"Filtered dataset from {original_len} to {len(data)} samples.")
            if not data:
                print(f"Warning: No data points found for dimension {self.filter_dim} in {folder_path}")

        cleaned_dataset = []
        for point in data:
            source = point['source']
            if np.any(np.isnan(source)) or np.any(np.isinf(source)):
                continue
            cleaned_dataset.append(point)

        return MetaStatDataset(cleaned_dataset)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training, validation, and testing.

        Args:
            stage: Current stage (fit, validate, test)
        """
        # Load datasets
        self.train_data = self._load_dataset(self.train_folder)
        self.val_data = self._load_dataset(self.val_folder)

        if self.test_folder:
            self.test_data = self._load_dataset(self.test_folder)
        else:
            self.test_data = self.val_data

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a dataloader for the given dataset."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=lambda batch: collate_fn(batch, target_feature_dim=self.target_feature_dim),
            pin_memory=True
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return self._create_dataloader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return self._create_dataloader(self.val_data)

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return self._create_dataloader(self.test_data)