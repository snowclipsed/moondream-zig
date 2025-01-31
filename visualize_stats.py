import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math


def visualize(filename, output):
    if not os.path.isfile(filename):
        print(f"Error: '{filename}' does not exist.")
        return

    df = pd.read_csv(filename)
    df["Sequence"] = range(1, len(df) + 1)
    numeric_cols = (
        df.select_dtypes(include=[np.number])
        .columns.difference(["Timestamp", "Sequence"])
        .tolist()
    )

    cols, width_per_col, height_per_row = 4, 8, 5
    rows = math.ceil(len(numeric_cols) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(width_per_col * cols, height_per_row * rows)
    )

    axes = axes.flatten() if rows * cols > 1 else [axes]
    for i, col in enumerate(numeric_cols):
        axes[i].plot(df["Sequence"], df[col], marker="o", linestyle="-")
        axes[i].set(
            title=col, xlabel="Sequence", ylabel=col, ylim=(0, df[col].max() * 1.1)
        )
        axes[i].grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output, dpi=300)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualise_stats.py <filename.csv> <stats.png>")
    else:
        output = (
            sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.curdir, "stats.png")
        )
        visualize(sys.argv[1], output)
