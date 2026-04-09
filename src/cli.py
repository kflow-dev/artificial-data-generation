from pathlib import Path
import subprocess
import sys

import click
import pandas as pd
import pytest

from .analysis import calculate_kl_divergence, recover_parameters
from .generators import get_generator
from .viz import plot_comparison
from .workflows import (
    compare_numeric_columns,
    ensure_parent_dir,
    impute_vip_data,
    run_generation_workflow,
)


@click.group()
def cli():
    """Artificial Data Generation Framework CLI."""
    pass


@cli.command()
@click.option("--method", type=click.Choice(["random", "specialized", "gan"]), required=True)
@click.option("--samples", default=1000, type=int)
@click.option("--output", default="data/synthetic/data.csv")
@click.option("--random-state", default=42, type=int)
def generate(method, samples, output, random_state):
    """Generate synthetic VIP customer data using a specified engine."""
    gen = get_generator(method)
    df = gen.generate(samples, random_state=random_state)
    output_path = ensure_parent_dir(output)
    df.to_csv(output_path, index=False)
    click.echo(f"✅ Generated {samples} VIP records using {method}; saved to {output_path}")


@cli.command()
@click.option("--samples", default=1000, type=int)
@click.option("--raw-output", default="data/raw/vip_seed.csv")
@click.option("--checkpoint-output", default="data/checkpoints/vip_seed.csv")
@click.option("--random-state", default=42, type=int)
def bootstrap(samples, raw_output, checkpoint_output, random_state):
    """Create notebook-ready raw and checkpoint VIP datasets."""
    outputs = run_generation_workflow(
        samples=samples,
        random_state=random_state,
        raw_output=raw_output,
        checkpoint_output=checkpoint_output,
    )
    click.echo(
        "✅ Created notebook datasets: "
        f"raw={raw_output} ({len(outputs['raw'])} rows), "
        f"checkpoint={checkpoint_output} ({len(outputs['checkpoint'])} rows)"
    )


@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--output", default="data/synthetic/imputed.csv")
def impute(input_path, output):
    """Impute missing values for VIP customer datasets."""
    df = pd.read_csv(input_path)
    imputed_df = impute_vip_data(df)
    output_path = ensure_parent_dir(output)
    imputed_df.to_csv(output_path, index=False)
    click.echo(f"✅ Imputation complete. Saved to {output_path}")


@cli.command()
@click.option("--real", required=True)
@click.option("--synth", required=True)
@click.option("--column", default="avg_transaction", show_default=True)
@click.option("--plot-output", default="data/synthetic/comparison.png", show_default=True)
def analyze(real, synth, column, plot_output):
    """Perform fidelity analysis and Bayesian recovery for a chosen column."""
    df_r = pd.read_csv(real)
    df_s = pd.read_csv(synth)

    kl = calculate_kl_divergence(df_r[column].dropna().values, df_s[column].dropna().values)
    click.echo(f"📊 KL Divergence ({column}): {kl:.4f}")

    click.echo("🧬 Running Bayesian parameter recovery...")
    trace = recover_parameters(df_s, column=column)
    click.echo(f"✅ Posterior Mean ({column}): {trace.posterior['mu'].mean().values:.4f}")

    metrics = compare_numeric_columns(df_r, df_s)
    if not metrics.empty:
        click.echo("\nComparison summary:")
        click.echo(metrics.to_string(index=False))

    plot_path = ensure_parent_dir(plot_output)
    plot_comparison(df_r, df_s, str(plot_path), column=column)
    click.echo(f"🖼️ Comparison plot saved to {plot_path}")


NOTEBOOKS = [
    "notebooks/s00_data_pipeline.ipynb",
    "notebooks/s01_data_generation.ipynb",
    "notebooks/s02_data_preprocessing.ipynb",
    "notebooks/s03_data_analysis.ipynb",
    "notebooks/s04_data_visualization.ipynb",
]


def _run_command(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise click.ClickException(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


@cli.command(name="run-notebook")
@click.argument("notebook", required=False)
@click.option("--all", "run_all", is_flag=True, help="Run all project notebooks in sequence.")
@click.option("--inplace", is_flag=True, help="Execute notebooks in place.")
@click.option("--output-dir", default=None, help="Directory for executed notebook copies.")
def run_notebook(notebook, run_all, inplace, output_dir):
    """Execute one notebook or all notebooks via jupyter nbconvert."""
    selected = NOTEBOOKS if run_all else [notebook]
    if not run_all and not notebook:
        raise click.ClickException("Provide NOTEBOOK or use --all.")

    for nb in selected:
        command = ["jupyter", "nbconvert", "--to", "notebook", "--execute", nb]
        if inplace:
            command.append("--inplace")
        if output_dir:
            ensure_parent_dir(Path(output_dir) / "placeholder.txt")
            command.extend(["--output-dir", output_dir])
        click.echo(f"📓 Running: {nb}")
        _run_command(command)


@cli.command(name="export-notebooks")
@click.option("--output-dir", default="notebooks/python", show_default=True)
def export_notebooks(output_dir):
    """Convert all project notebooks to Python scripts."""
    ensure_parent_dir(Path(output_dir) / "placeholder.txt")
    for nb in NOTEBOOKS:
        command = ["jupyter", "nbconvert", "--to", "python", nb, "--output-dir", output_dir]
        click.echo(f"📝 Exporting: {nb}")
        _run_command(command)


@cli.command()
@click.option("--suite", default="all")
def test(suite):
    """Run the project test suite."""
    click.echo(f"🧪 Running {suite} tests...")
    retcode = pytest.main([f"tests/test_{suite}.py" if suite != "all" else "tests/"])
    sys.exit(retcode)


if __name__ == "__main__":
    cli()
