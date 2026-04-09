import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_comparison(real: pd.DataFrame, synthetic: pd.DataFrame, filename: str, column: str | None = None):
    target_column = column or real.columns[0]
    plt.figure(figsize=(10, 6))
    sns.kdeplot(real[target_column].dropna(), label="Real/Seed", color="blue")
    sns.kdeplot(synthetic[target_column].dropna(), label="Synthetic", color="red")
    plt.title(f"Distribution Comparison: {target_column}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
