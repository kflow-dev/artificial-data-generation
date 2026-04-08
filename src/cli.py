import click
import pandas as pd
from .generators import get_generator
from .processing import bayesian_impute, introduce_nans
from .analysis import calculate_kl_divergence, recover_parameters
from .viz import plot_comparison
import pytest
import sys

@click.group()
def cli():
    """Artificial Data Generation Framework CLI"""
    pass

@cli.command()
@click.option('--method', type=click.Choice(['random', 'specialized', 'gan']), required=True)
@click.option('--samples', default=1000, type=int)
@click.option('--output', default='data/synthetic/data.csv')
def generate(method, samples, output):
    """Generate synthetic data using a specified engine."""
    gen = get_generator(method)
    df = gen.generate(samples)
    df.to_csv(output, index=False)
    click.echo(f"✅ Generated {samples} samples using {method} saved to {output}")

@cli.command()
@click.option('--input', required=True)
@click.option('--output', default='data/synthetic/imputed.csv')
def impute(input, output):
    """Impute missing values using Bayesian Iterative Imputation."""
    df = pd.read_csv(input)
    # If no NaNs, introduce some for the sake of the demo
    if df.isnull().sum().sum() == 0:
        df = introduce_nans(df)
    
    imputed_df = bayesian_impute(df)
    imputed_df.to_csv(output, index=False)
    click.echo(f"✅ Imputation complete. Saved to {output}")

@cli.command()
@click.option('--real', required=True)
@click.option('--synth', required=True)
def analyze(real, synth):
    """Perform statistical analysis and Bayesian recovery."""
    df_r = pd.read_csv(real)
    df_s = pd.read_csv(synth)
    
    kl = calculate_kl_divergence(df_r.iloc[:, 0].values, df_s.iloc[:, 0].values)
    click.echo(f"📊 KL Divergence: {kl:.4f}")
    
    click.echo("🧬 Running Bayesian Parameter Recovery...")
    trace = recover_parameters(df_s)
    click.echo(f"✅ Posterior Mean: {trace.posterior['mu'].mean().values:.4f}")
    
    plot_comparison(df_r, df_s, "data/synthetic/comparison.png")
    click.echo("🖼️ Comparison plot saved to data/synthetic/comparison.png")

@cli.command()
@click.option('--suite', default='all')
def test(suite):
    """Run the project test suite."""
    click.echo(f"🧪 Running {suite} tests...")
    retcode = pytest.main([f"tests/test_{suite}.py" if suite != 'all' else "tests/"])
    sys.exit(retcode)

if __name__ == "__main__":
    cli()
