from __future__ import annotations
import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from omegaconf import OmegaConf, DictConfig
from mist_statinf.utils.logging import setup_logging
from mist_statinf.utils.set_seed import set_seed
from mist_statinf.data.meta_dataset import MIRegressionGenerator        

def apply_subplot_style():
    plt.tick_params(axis='both', which='both', labelsize=13, direction='in', top=True, right=True)
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

def plot_mi_distributions(data, output_filename=None):
    sns.set_style("ticks")
    plt.rcParams.update({
        "font.family":"serif","font.serif":["Times New Roman"],"mathtext.fontset":"stix",
        "axes.spines.top":True,"axes.spines.right":True,
        "axes.titleweight":"bold","axes.titlepad":20,
        "axes.labelsize":16,"xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,"figure.titlesize":16
    })
    df = pd.DataFrame(data)
    plt.figure(figsize=(14,10))
    plt.subplot(1,2,1)
    df['target'].plot(kind='hist', title='Distribution of Target', xlabel='Target Value',
                      ylabel='Frequency', color='tab:green', edgecolor='black', alpha=0.7)
    apply_subplot_style()
    plt.subplot(1,2,2)
    df['distribution'].value_counts().plot(kind='bar', color='tab:blue', edgecolor='black', alpha=0.7)
    plt.title('Number of Entries per Distribution'); plt.xticks(rotation=45, ha='right')
    plt.xlabel('Distribution Type'); plt.ylabel('Count'); apply_subplot_style()
    plt.tight_layout()
    if output_filename: plt.savefig(output_filename, bbox_inches="tight")
    else: plt.show()

def plot_distributions(data, output_filename=None):
    sns.set_style("ticks")
    plt.rcParams.update({
        "font.family":"serif","font.serif":["Times New Roman"],"mathtext.fontset":"stix",
        "axes.spines.top":True,"axes.spines.right":True,"axes.titleweight":"bold","axes.titlepad":20,
        "axes.labelsize":16,"xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,"figure.titlesize":16
    })
    df = pd.DataFrame(data)
    plt.figure(figsize=(14,10))
    # 1
    plt.subplot(2,2,1)
    df['distribution'].value_counts().plot(kind='bar', color='tab:blue', edgecolor='black', alpha=0.7)
    plt.title('Number of Entries per Distribution'); plt.xticks(rotation=45); plt.xlabel(''); plt.ylabel('Count')
    apply_subplot_style()
    # 2
    mean_difficulty = df.groupby('distribution')['np_std_difficulty'].mean().sort_values(ascending=False)

    plt.subplot(2,2,2)
    sns.boxplot(x='distribution', y='np_std_difficulty', data=df, color='tab:blue',
                boxprops={"alpha":0.7}, order=mean_difficulty.index)
    plt.title('np.std Difficulty per Distribution'); plt.xlabel('Distribution'); plt.ylabel('np.std Difficulty')
    plt.xticks(rotation=45); apply_subplot_style()
    # 3
    plt.subplot(2,2,3)
    sns.histplot(df['np_std_difficulty'].dropna(), kde=True, color='orange', bins=10)
    plt.title('Distribution of np.std Difficulty Across All Distributions'); plt.xlabel('np.std Difficulty')
    apply_subplot_style()
    # 4
    plt.subplot(2,2,4)
    corr = df[['excess_kurtosis','np_std_difficulty']].dropna().corr().iloc[0,1]
    sns.scatterplot(x=df['excess_kurtosis'], y=df['np_std_difficulty'], color="purple", alpha=0.7)
    plt.title(f'Excess Kurtosis vs np.std Difficulty (Corr: {corr:.2f})')
    plt.xlabel('Excess Kurtosis'); plt.ylabel('np.std Difficulty'); plt.xscale('log'); plt.yscale('log')
    apply_subplot_style()
    plt.tight_layout()
    if output_filename: plt.savefig(output_filename, bbox_inches="tight")
    else: plt.show()

def generate_main(config_path: str, version: str = ""):
    """
    Generate a synthetic mutual information (MI) meta-dataset and optionally produce summary plots.

    Args:
        config_path (str):  
            Path to a YAML configuration file.  
            The configuration must define:
            - experiment.name (str): Name of the experiment.
            - experiment.log_file (str, optional): Log file name. Defaults to "experiment.log".
            - experiment.seed (int, optional): Random seed. Defaults to 1234.
            - number_meta_datapoints (int): Number of meta-datasets to generate.
            - n_row_range (tuple[int, int]): Range of sample sizes for each dataset.
            - meta_distribution (dict): Meta-distribution definition (serializable via OmegaConf).
            - n_dim (int): Dimensionality of the generated data.
            - grid (bool, optional): If True, uses grid search parameters.
            - grid_params (dict, optional): Parameters for grid search, required if `grid=True`.
            - plot_stats (bool, optional): Whether to generate summary plots. Defaults to True.
            - experiment.type (str, optional): If "MI", calls `plot_mi_distributions`.

        version (str, optional):  
            Optional suffix appended to the output folder name.  
            Useful to distinguish runs with the same experiment name. Defaults to "".

    Returns:
        None.  
        Saves generated datasets, configuration, logs, and optional plots to:
        `data/<experiment_name>[_<version>][_grid]/`

    Example:
        >>> generate_main("configs/data_generation/train.yaml", version="v1")

    """
    cfg: DictConfig = OmegaConf.load(config_path)
    name = cfg['experiment']['name']
    fname = name + (f"_{version}" if version else "")
    if cfg.get("grid"): fname += "_grid"
    output_dir = os.path.join("data", fname)

    logger = setup_logging(output_dir, cfg['experiment'].get('log_file',"experiment.log"))
    seed = cfg['experiment'].get("seed", 1234); set_seed(seed); logger.info(f"Seeds set: {seed}")

    OmegaConf.save(cfg, os.path.join(output_dir, f"{name}.yaml"))
    logger.info("Loaded configuration:\n" + OmegaConf.to_yaml(cfg))

    gen = MIRegressionGenerator(
        number_meta_datapoints=cfg['number_meta_datapoints'],
        n_row_range=tuple(cfg['n_row_range']),
        meta_distribution=OmegaConf.to_container(cfg['meta_distribution'], resolve=True),
        n_dim=cfg['n_dim'],
        output_folder=output_dir
    )

    grid_params = cfg.grid_params if cfg.get("grid") else None
    summary = gen.generate(grid_params)

    if cfg.get("plot_stats", True):
        plot_filename = os.path.join(output_dir, "summary_stats_plots.pdf")
        if cfg.experiment.get("type","") == "MI":
            plot_mi_distributions(summary["datasets"], plot_filename)
        logger.info(f"Summary plots saved to: {plot_filename}")

    logger.info(f"Dataset saved to: {output_dir}")
