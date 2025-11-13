import typer
from typing import Optional, Dict, Tuple
from pathlib import Path

app = typer.Typer(add_completion=False, help="MIST — neural MI estimator with training, inference, and tools.")


@app.command("train")
def train(
    config: str = typer.Argument(..., help="Path to YAML config for training"),
    ckpt_out: str = typer.Option("checkpoints", help="(Optional) target dir for checkpoints (informational)"),
):
    """Train a MIST model with PyTorch Lightning."""
    from mist_statinf.train.train import train_main
    train_main(config, ckpt_out)


@app.command("infer")
def infer(
    config: str = typer.Argument(..., help="Path to YAML config for inference"),
    ckpt_dir: str = typer.Argument(..., help="Run folder with checkpoints (e.g., logs/<exp>/run_YYYYmmdd-HHMMSS)"),
    out_path: str = typer.Option("mi_results.json", help="Path to save JSON summary"),
):
    """Run inference for a trained MIST model (bootstrap / qcqr_calib modes)."""
    from mist_statinf.infer.inference import infer_main
    infer_main(config, ckpt_dir, out_path)


@app.command("baselines")
def baselines(
    config: str = typer.Argument(..., help="Path to YAML config for baselines evaluation"),
):
    """Evaluate classic MI baselines (KSG, MINE, InfoNCE, NWJ, CCA) on a meta-dataset."""
    from mist_statinf.train.baselines import baselines_main
    baselines_main(config)


@app.command("generate")
def generate(
    config: str = typer.Argument(..., help="Path to YAML config for meta-dataset generation"),
    version: str = typer.Option("", help="Suffix for output dataset folder name"),
):
    """Generate synthetic meta-datasets."""
    from mist_statinf.data.generate import generate_main
    generate_main(config, version)


@app.command("tune")
def tune_command(
    run_dir: str = typer.Argument(..., help="Path to a previous training run dir (logs/<exp>/run_YYYYmmdd-HHMMSS)"),
    model_type: str = typer.Option("QCQR", "--model-type", "-m", help="Override loss type: MSE or QCQR"),
    n_trials: int = typer.Option(50, "--n-trials", "-n", help="Number of Optuna trials"),
    timeout_sec: Optional[int] = typer.Option(
        None, "--timeout-sec", "-t", help="Optional wall-clock timeout in seconds"
    ),
):
    from mist_statinf.train.hparams_search import hparam_main
    hparam_main(run_dir=run_dir, model_type=model_type, n_trials=n_trials, timeout_sec=timeout_sec)


@app.command("get-data")
def get_data_command(
    preset: Optional[str] = typer.Option(
        None, "--preset", "-p",
        help="Named preset of dataset parts (e.g., 'mini', 'full')."
    ),
    target_dir: Path = typer.Option(
        Path("data"), "--dir", "-t",
        help="Directory to place downloaded files."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite",
        help="Re-download even if the file exists."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Reduce output verbosity."
    ),
):
    from mist_statinf.utils.downloads import fetch_files, preset_bundle
    items: Dict[str, Tuple[str, Optional[str]]] = {}

    if preset:
        items.update(preset_bundle(preset))

    fetch_files(items=items, target_dir=target_dir, quiet=quiet, overwrite=overwrite)
    if not quiet:
        typer.echo(f"✔ Done. Files are in: {target_dir.resolve()}")



@app.command("version")
def version():
    """Show package version."""
    try:
        from importlib.metadata import version as _v
        typer.echo(_v("mist"))
    except Exception:
        typer.echo("mist (dev)")


def main():
    app()

