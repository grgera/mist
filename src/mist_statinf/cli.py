import typer

app = typer.Typer(add_completion=False, help="MIST — neural MI estimator with training, inference, and tools.")

# ------------------------- Commands -------------------------

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
    """Run inference for a trained MIST model (point / bootstrap / qcqr_calib modes)."""
    from mist_statinf.infer.inference import infer_main
    infer_main(config, ckpt_dir, out_path)


@app.command("baselines")
def baselines(
    config: str = typer.Argument(..., help="Path to YAML config for baselines evaluation"),
):
    """Evaluate classic MI baselines (KSG, MINE, InfoNCE, NWJ, CCA) on a meta-dataset."""
    from mist_statinf.train.run_baselines import baselines_main
    baselines_main(config)


@app.command("generate")
def generate(
    config: str = typer.Argument(..., help="Path to YAML config for meta-dataset generation"),
    version: str = typer.Option("", help="Suffix for output dataset folder name"),
):
    """Generate synthetic meta-datasets (with optional plots)."""
    from mist_statinf.data.generate import generate_main
    generate_main(config, version)


@app.command("tune")
def tune(
    model_type: str = typer.Option("QCQR", help="Model family to tune: 'QCQR' or 'MSE'"),
    n_trials: int = typer.Option(50, help="Number of Optuna trials"),
):
    """Hyperparameter search with Optuna."""
    from mist_statinf.train.hparam_search import hparam_main
    hparam_main(model_type=model_type, n_trials=n_trials)


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

