"""Audit a Label Studio export for the rate of ``unexpected_case`` QC flags per dataset.

Given a Label Studio export CSV, report how many labeled
cells each source dataset contributed and what fraction were flagged
``unexpected_case = yes``. A high rate means that dataset's cells for the class being
labeled are unreliable and may warrant exclusion.

Usage:
    python scripts/audit_unexpected.py path/to/export.csv
"""

from __future__ import annotations

import argparse

import pandas as pd


def audit(
    df: pd.DataFrame,
    flag_column: str = "unexpected_case",
    flag_value: str = "yes",
    group_column: str = "source_dataset",
) -> pd.DataFrame:
    """Per-dataset table of total labeled, flagged count, and flagged ratio.

    Args:
        df: a loaded Label Studio export.
        flag_column: the QC column whose value marks a flagged cell.
        flag_value: the value in ``flag_column`` that counts as flagged.
        group_column: the column to group by (the source dataset).

    Returns:
        A DataFrame indexed by dataset with ``total``, ``flagged``, and ``ratio``
        columns, sorted by ``ratio`` descending (worst offenders first).

    Raises:
        KeyError: if an expected column is missing from the export.
    """
    for col in (flag_column, group_column):
        if col not in df.columns:
            raise KeyError(
                f"expected column {col!r} not in export; found {list(df.columns)}"
            )

    total = df[group_column].value_counts()
    flagged = df[df[flag_column] == flag_value][group_column].value_counts()
    # reindex so datasets with zero flags still show up as 0
    flagged = flagged.reindex(total.index, fill_value=0)
    ratio = flagged / total

    report = pd.DataFrame({"total": total, "flagged": flagged, "ratio": ratio})
    return report.sort_values("ratio", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_csv", help="path to the Label Studio export CSV")
    parser.add_argument("--flag-column", default="unexpected_case")
    parser.add_argument("--flag-value", default="yes")
    parser.add_argument("--group-column", default="source_dataset")
    args = parser.parse_args()

    df = pd.read_csv(args.export_csv)
    report = audit(df, args.flag_column, args.flag_value, args.group_column)
    print(report.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
