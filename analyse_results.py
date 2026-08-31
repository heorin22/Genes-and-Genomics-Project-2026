import argparse
from pathlib import Path

import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd


CLASS_ORDER = [
    "Benign",
    "Class IV",
    "Class III",
    "Class II",
]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Statistical analysis of ESM-2 variant scores."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV produced by score_variants.py"
    )

    parser.add_argument(
        "--stats-output",
        default="descriptive_statistics.csv"
    )

    parser.add_argument(
        "--tukey-output",
        default="tukey_results.csv"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    df = pd.read_csv(args.input)

    required_columns = {
        "Mutation",
        "Class",
        "LLR"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Input must contain Mutation, Class and LLR columns."
        )

    # ---------------------------------
    # Descriptive statistics
    # ---------------------------------

    summary = (
        df.groupby("Class")["LLR"]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ])
    )

    existing_classes = [
        cls
        for cls in CLASS_ORDER
        if cls in summary.index
    ]

    summary = summary.reindex(
        existing_classes
    )

    stats_output = Path(
        args.stats_output
    )

    stats_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(stats_output)

    print("\nDescriptive statistics")
    print("----------------------")
    print(summary)

    # ---------------------------------
    # Tukey HSD
    # ---------------------------------

    tukey = pairwise_tukeyhsd(
        endog=df["LLR"],
        groups=df["Class"],
        alpha=0.05
    )

    tukey_df = pd.DataFrame(
        tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    tukey_output = Path(
        args.tukey_output
    )

    tukey_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tukey_df.to_csv(
        tukey_output,
        index=False
    )

    print("\nTukey HSD")
    print("---------")
    print(tukey)


if __name__ == "__main__":
    main()