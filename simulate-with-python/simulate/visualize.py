import logging

# Disable debug logs for matplotlib (very verbose)
rootlogger = logging.getLogger()
origlevel = rootlogger.level
rootlogger.setLevel(level="INFO")

# get new logger for this module only
logger = logging.getLogger(__name__)

import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="ticks")

# Reset original log level for the root level
rootlogger.setLevel(origlevel)


def from_old_format(df: pd.DataFrame):
    logger.info(f"Converting this dataframe to new format: \n{df}")
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

    if "checks" in df.columns:
        df = from_old_format(df)

    # Round oracle_calls to nearest multiple of 200
    df = round_stride_of_type(df, "oracle_calls", 500)
    df = round_stride_of_type(df, "unsatisfied", 20)

    # Remove "oracle calls" and "unsatisfied", they are not interesting here
    # df = df[~df["stride_type"].isin(['oracle_calls', 'unsatisfied'])]
    df = df[df["weight"].isin([50])]

    logger.info("Plotting data")
    g = sns.relplot(
        data=df,
        x="stride",
        y="count",
        col="count_type",
        row="stride_type",
        #hue="label",
        kind="line",
        facet_kws={"margin_titles": True, "sharey": False, "sharex": False},
    )
    g.set_axis_labels("Number of ... (see right margin)", "Count of... (see top margin)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend()
    #g.tight_layout()

    logger.info("Writing data to file")
    plt.savefig("view_hqc_simulation_csv.pdf")

def round_stride_of_type(df, stride_type, multiple_of):
    cond = df["stride_type"] == stride_type
    half = multiple_of//2
    df.loc[cond,"stride"] = df.loc[cond,"stride"].apply(lambda s: (s+half) / multiple_of).astype(int) * 100
    return df
