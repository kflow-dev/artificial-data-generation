import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_comparison(real: pd.DataFrame, synthetic: pd.DataFrame, filename: str):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(real.iloc[:, 0], label='Real/Seed', color='blue')
    sns.kdeplot(synthetic.iloc[:, 0], label='Synthetic', color='red')
    plt.title("Distribution Comparison")
    plt.legend()
    plt.savefig(filename)
    plt.close()
