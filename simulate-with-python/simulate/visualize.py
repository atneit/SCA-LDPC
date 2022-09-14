import logging

# Disable debug logs for matplotlib (very verbose)
rootlogger = logging.getLogger()
origlevel = rootlogger.level
rootlogger.setLevel(level="INFO")

# get new logger for this module only
logger = logging.getLogger(__name__)

import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="ticks")

# Reset original log level for the root level
rootlogger.setLevel(origlevel)


def wide_to_long_format(df: pd.DataFrame):
    logger.info(f"Converting this DataFrame from wide to long: \n{df}")
    meta_col = ["label", "alg", "weight", "epsilon0", "epsilon1"]
    new_df = pd.DataFrame(
        columns=meta_col + ["stride_type", "stride", "count_type", "count"]
    )
    for old_stride in ["checks", "oracle_calls", "unsatisfied"]:
        for old_count in [
            "good_flips",
            "bad_flips",
            "found_bad_satisfied_checks",
            "found_bad_unsatisfied_checks",
            "remaining-flips",
            "found_bad_checks",
        ]:
            new_df = extract_pair(df, meta_col, new_df, old_stride, old_count)

    to_convert = ["label", "alg", "stride_type", "count_type", "success"]
    new_df[to_convert] = new_df[to_convert].astype("category")
    to_convert = ["weight", "stride", "count"]
    new_df[to_convert] = new_df[to_convert].astype("int")

    logger.info(f"new_df: \n{new_df.dtypes}")
    logger.info(f"new_df: \n{new_df}")
    return new_df


def extract_pair(
    df: pd.DataFrame, meta_col, new_df: pd.DataFrame, old_stride, old_count
):
    filtered = df[meta_col + [old_stride, old_count, "success"]]
    filtered = filtered.rename(columns={old_stride: "stride", old_count: "count"})
    filtered["stride_type"] = old_stride
    filtered["count_type"] = old_count
    new_df = pd.concat([new_df, filtered])
    return new_df


def view_hqc_simulation_csv(csv_file):
    logger.info(f"Reading file: {csv_file}")
    df = pd.read_csv(csv_file)

    # Combine bit flips to calculate number of correct bit positions, as a percentage
    max_flips = df["good_flips"].max()
    df["remaining-flips"] = max_flips + df["bad_flips"] - df["good_flips"]
    df["found_bad_checks"] = (
        df["found_bad_unsatisfied_checks"] + df["found_bad_satisfied_checks"]
    )

    df = wide_to_long_format(df)

    # Round oracle_calls to nearest multiple of 200
    df = round_stride_of_type(df, "oracle_calls", 500)
    df = round_stride_of_type(df, "unsatisfied", 20)

    # Remove "oracle calls" and "unsatisfied", they are not interesting here
    # Filter away uninteresting data
    df = df[
        (
            (True)
            & (df["weight"].isin([50]))
            & (df["stride_type"].isin(["checks", "oracle_calls"]))
            & (df["count_type"].isin(["remaining-flips", "found_bad_checks"]))
        )
    ]

    # Rename for human readable axis labels
    df["stride_type"] = df["stride_type"].cat.rename_categories(
        {
            "checks": "parity checks",
            "oracle_calls": "oracle calls",
            "unsatisfied": "unsatisfied parity checks",
        }
    )
    df["count_type"] = df["count_type"].cat.rename_categories(
        {
            "good_flips": "correct bit-flips",
            "bad_flips": "incorrect bit-flips",
            "found_bad_unsatisfied_checks": "found_bad_unsatisfied_checks",
            "found_bad_satisfied_checks": "found_bad_satisfied_checks",
            "remaining-flips": "remaining bit-flips",
            "found_bad_checks": "detected bad parity checks",
        }
    )

    df["stride_type"] = df["stride_type"].cat.remove_unused_categories()
    df["count_type"] = df["count_type"].cat.remove_unused_categories()

    logger.info(
        "stride_types remaining: " + str(list(df["stride_type"].cat.categories))
    )
    logger.info("count_types remaining: " + str(list(df["count_type"].cat.categories)))

    logger.info("Plotting data")
    g = sns.relplot(
        data=df,
        x="stride",
        y="count",
        col="count_type",
        row="stride_type",
        hue="epsilon1",
        kind="line",
        facet_kws={"margin_titles": False, "sharey": False, "sharex": False},
    )
    for ((row, col), ax) in g.axes_dict.items():
        if col in ["remaining bit-flips"]:
            columns=["stride", "epsilon1"]
            success_stride = pd.DataFrame(columns=columns)
            ax2 = ax.twinx()
            ax2.tick_params(axis="y", right=False)
            mask = (
                (df["count_type"] == col)
                & (df["stride_type"] == row)
                & (df["success"] == True)
            )
            success_stride = df.loc[mask,columns].reset_index(drop=True)
            print(success_stride.dtypes)
            print(success_stride)
            box = sns.boxplot(
                data=success_stride,
                ax=ax2,
                x="stride",
                y="epsilon1",
                hue="epsilon1",
                orient="h",
                dodge=True,
                width=0.05,
                notch=False,
                whis=(0, 100),
                # color="white",
                # saturation=0,
                # boxprops={'facecolor':'none', 'edgecolor':'black'},
                # medianprops={'color':'black'},
                # whiskerprops={'color':'black'},
                # capprops={'color':'black'}
            )
            logger.info(f"Added boxplot to ({row},{col})")
            lines = box.get_lines()
            boxes = [c for c in box.get_children() if type(c).__name__ == "PathPatch"]
            lines_per_box = int(len(lines) / len(boxes))
            for median in lines[4 : len(lines) : lines_per_box]:
                x, y = (data.mean() for data in median.get_data())
                box.annotate(
                    "Decoding success",
                    xy=(x, y),
                    xycoords="data",
                    xytext=(30, 30),
                    textcoords="offset pixels",
                    horizontalalignment='center',
                    verticalalignment='top',
                    clip_on=True,
                )
    g.set_axis_labels(
        "Number of", "Number of"
    )
    g.set_titles(col_template="y = {col_name}", row_template="x = {row_name}")
    # g.add_legend()
    # g.tight_layout()

    logger.info("Writing data to file")
    g.savefig("view_hqc_simulation_csv.pdf")


def round_stride_of_type(df, stride_type, multiple_of):
    cond = df["stride_type"] == stride_type
    half = multiple_of // 2
    df.loc[cond, "stride"] = (
        df.loc[cond, "stride"].apply(lambda s: (s + half) / multiple_of).astype(int)
        * multiple_of
    )
    return df
