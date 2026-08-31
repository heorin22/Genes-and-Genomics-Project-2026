import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind


CLASS_ORDER = [
    "Benign",
    "Class II",
    "Class III",
    "Class IV",
]


COMPARISONS = [
    ("Benign", "Class II"),
    ("Benign", "Class III"),
    ("Class II", "Class IV"),
    ("Class III", "Class IV"),
]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Plot ESM-2 CFTR variant scores."
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        default="cftr_llr_boxplot.png"
    )

    return parser.parse_args()


def significance_symbol(p_value):

    if p_value >= 0.05:
        return "ns"

    if p_value >= 0.01:
        return "*"

    if p_value >= 0.001:
        return "**"

    return "***"


def add_stat_annotations(
    ax,
    data,
    comparisons,
    order
):

    y_min = data["LLR"].min()
    y_max = data["LLR"].max()

    y_range = y_max - y_min

    for i, (group1, group2) in enumerate(comparisons):

        group1_values = data.loc[
            data["Class"] == group1,
            "LLR"
        ]

        group2_values = data.loc[
            data["Class"] == group2,
            "LLR"
        ]

        if (
            len(group1_values) == 0
            or len(group2_values) == 0
        ):
            continue

        _, p_value = ttest_ind(
            group1_values,
            group2_values,
            equal_var=False
        )

        symbol = significance_symbol(
            p_value
        )

        x1 = order.index(group1)
        x2 = order.index(group2)

        y = (
            y_max
            + y_range * 0.10
            + i * y_range * 0.15
        )

        height = y_range * 0.04

        ax.plot(
            [x1, x1, x2, x2],
            [y, y + height, y + height, y],
            linewidth=1.5,
            color="black"
        )

        ax.text(
            (x1 + x2) / 2,
            y + height,
            symbol,
            ha="center",
            va="bottom"
        )


def main():

    args = parse_args()

    df = pd.read_csv(args.input)

    order = [
        cls
        for cls in CLASS_ORDER
        if cls in df["Class"].unique()
    ]

    plt.figure(
        figsize=(9, 7)
    )

    ax = sns.boxplot(
        data=df,
        x="Class",
        y="LLR",
        order=order,
        hue="Class",
        palette="viridis",
        linewidth=1.5,
        legend=False
    )

    sns.stripplot(
        data=df,
        x="Class",
        y="LLR",
        order=order,
        color="black",
        alpha=0.6,
        jitter=True,
        ax=ax
    )

    valid_comparisons = [
        pair
        for pair in COMPARISONS
        if pair[0] in order
        and pair[1] in order
    ]

    add_stat_annotations(
        ax,
        df,
        valid_comparisons,
        order
    )

    ax.axhline(
        0,
        linestyle="--",
        alpha=0.5
    )

    ax.set_title(
        "ESM-2 Pathogenicity Prediction "
        "by CFTR Mutation Class"
    )

    ax.set_xlabel(
        "Functional Class"
    )

    ax.set_ylabel(
        "ESM-2 Log-Likelihood Ratio (LLR)\n"
        "Lower = More Damaging"
    )

    plt.tight_layout()

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Saved plot to {output}"
    )


if __name__ == "__main__":
    main()
