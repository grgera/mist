import logging
import os
import random
import torch
import numpy as np


def reset_logger(logger):
    """Remove all handlers from the given logger."""
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def setup_logging(output_dir, log_file):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_file)

    # Create a custom logger
    logger = logging.getLogger()

    # Reset the logger to avoid duplication of handlers
    reset_logger(logger)

    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_seeds(seed):
    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for numpy
    np.random.seed(seed)

    # Set the seed for PyTorch CPU and GPU (if available)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Ensure deterministic behavior for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def bootstrap_data(dataset, n_resamples):
    n = dataset.shape[0]
    resamples = []
    for _ in range(n_resamples):
        # sample indices with replacement
        idx = np.random.choice(n, size=n, replace=True)
        resamples.append(dataset[idx])
    return resamples


