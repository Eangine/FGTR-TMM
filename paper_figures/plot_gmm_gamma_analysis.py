import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fmt_value(x):
    """Format numbers like paper figures: 89 instead of 89.0."""
    return f"{x:.1f}".rstrip("0").rstrip(".")


def load_data(csv_path=None):
    """Load gamma analysis data.

    CSV columns:
        gamma,i2t_r1,i2t_r5,i2t_r10,t2i_r1,t2i_r5,t2i_r10

    If csv_path is not provided, this function uses example values. Replace the
    arrays below with your real experimental results if you prefer editing this
    script directly.
    """
    if csv_path:
        required = [
            "gamma",
            "i2t_r1",
            "i2t_r5",
            "i2t_r10",
            "t2i_r1",
            "t2i_r5",
            "t2i_r10",
        ]
        rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file has no header.")
            missing = [c for c in required if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing required CSV columns: {missing}")
            for row in reader:
                rows.append({key: float(row[key]) for key in required})
        return {key: np.array([row[key] for row in rows], dtype=float) for key in required}

    # Example values from the reference style. Replace these with your results.
    gamma = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    data = {
        "gamma": gamma,
        "i2t_r1": [89.0, 89.3, 89.5, 89.9, 89.8, 89.4, 88.8, 88.8, 89.0],
        "i2t_r5": [97.6, 97.9, 98.0, 97.8, 97.3, 97.8, 98.0, 98.0, 97.7],
        "i2t_r10": [98.3, 98.8, 99.0, 98.7, 99.0, 99.1, 99.0, 98.6, 99.3],
        "t2i_r1": [73.0, 74.1, 74.2, 74.2, 74.9, 74.6, 74.3, 73.8, 74.2],
        "t2i_r5": [91.6, 92.2, 92.3, 92.4, 92.4, 92.3, 92.3, 92.4, 92.1],
        "t2i_r10": [94.8, 95.5, 95.7, 95.5, 95.6, 95.7, 95.7, 95.6, 95.6],
    }
    return {key: np.array(value, dtype=float) for key, value in data.items()}


def plot_gamma_analysis(df, output_dir, prefix, xlabel):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gamma = df["gamma"]
    metrics = [
        ("i2t_r1", "Image-to-Text R@1", "#fff1c9"),
        ("i2t_r5", "Image-to-Text R@5", "#ffe39a"),
        ("i2t_r10", "Image-to-Text R@10", "#ffd45a"),
        ("t2i_r1", "Text-to-Image R@1", "#dceccc"),
        ("t2i_r5", "Text-to-Image R@5", "#c6e1b6"),
        ("t2i_r10", "Text-to-Image R@10", "#a8d18d"),
    ]
    values = {name: df[name] for name, _, _ in metrics}
    rsum = np.sum([values[name] for name, _, _ in metrics], axis=0)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 14,
            "axes.linewidth": 1.2,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    x = np.arange(len(gamma))
    width = 0.42
    bottom = np.zeros_like(gamma, dtype=float)

    handles = []
    for name, label, color in metrics:
        bars = ax.bar(
            x,
            values[name],
            width,
            bottom=bottom,
            color=color,
            edgecolor=color,
            linewidth=0.8,
            label=label,
            zorder=2,
        )
        handles.append(bars[0])

        # Segment labels.
        for xi, btm, val in zip(x, bottom, values[name]):
            ax.text(
                xi,
                btm + val / 2,
                fmt_value(val),
                ha="center",
                va="center",
                fontsize=13,
                color="black",
                zorder=4,
            )
        bottom += values[name]

    # rSum line.
    line = ax.plot(
        x,
        rsum,
        color="#7b3dbb",
        linestyle="--",
        linewidth=1.6,
        marker="^",
        markersize=7,
        markerfacecolor="#e51b23",
        markeredgecolor="#006bb6",
        markeredgewidth=1.2,
        label="rSum",
        zorder=5,
    )[0]

    for xi, yi in zip(x, rsum):
        ax.text(
            xi,
            yi + 7,
            fmt_value(yi),
            ha="center",
            va="bottom",
            fontsize=13,
            color="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([fmt_value(v) for v in gamma], fontsize=15)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("rSum", fontsize=17)
    ax.set_ylim(0, max(600, float(rsum.max()) + 55))
    ax.set_xlim(-0.5, len(gamma) - 0.5)
    ax.tick_params(direction="in", length=4, width=1.1)
    ax.grid(False)

    # Two-row legend similar to common TMM/CVPR style figures.
    legend_handles = handles[:3] + handles[3:] + [line]
    legend_labels = [m[1] for m in metrics[:3]] + [m[1] for m in metrics[3:]] + ["rSum"]
    ax.legend(
        legend_handles,
        legend_labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.26),
        frameon=False,
        fontsize=13,
        handlelength=2.5,
        columnspacing=1.1,
        handletextpad=0.4,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.9])

    for ext in ("pdf", "svg", "png"):
        path = output_dir / f"{prefix}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GMM gamma parameter analysis.")
    parser.add_argument("--csv", default=None, help="Optional CSV file with gamma and recall metrics.")
    parser.add_argument("--output_dir", default="paper_figures", help="Directory for generated figures.")
    parser.add_argument("--prefix", default="gmm_gamma_analysis", help="Output file prefix.")
    parser.add_argument("--xlabel", default=r"$\gamma$", help="X-axis label, e.g., $\\gamma$ or $\\delta$.")
    args = parser.parse_args()

    df = load_data(args.csv)
    plot_gamma_analysis(df, args.output_dir, args.prefix, args.xlabel)


if __name__ == "__main__":
    main()
