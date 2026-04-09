from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_comparison(real: pd.DataFrame, synthetic: pd.DataFrame, filename: str, column: str | None = None):
    target_column = column or real.columns[0]
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(real[target_column].dropna(), label="Real/Seed", color="blue")
    sns.kdeplot(synthetic[target_column].dropna(), label="Synthetic", color="red")
    plt.title(f"Distribution Comparison: {target_column}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, filename: str, title: str = "Correlation Heatmap") -> None:
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
