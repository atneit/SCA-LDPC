GRID_WEIGHTS = False

import logging

# Disable debug logs for matplotlib (very verbose)
rootlogger = logging.getLogger()
origlevel = rootlogger.level
rootlogger.setLevel(level="INFO")

# get new logger for this module only
logger = logging.getLogger(__name__)

import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "lualatex",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.sans-serif": ["Computer Modern Sans Serif"],
        "font.monospace": ["Computer Modern Typewriter"],
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(
    style="darkgrid",
    font="serif",
    rc={
        "legend.frameon": True,
    },
)

# Reset original log level for the root level
rootlogger.setLevel(origlevel)


def plt_write(outputname="output.pgf"):
    w=4.8
    h=w if GRID_WEIGHTS else 2
    plt.gcf().set_size_inches(w=w, h=h)
    plt.tight_layout()
    plt.savefig(outputname, bbox_inches="tight")
    print("Output printed to " + outputname)
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def wide_to_long_format(df: pd.DataFrame):
    logger.info(f"Converting DataFrame from wide to long")
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


def load_data(csv_file):
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

    logger.info(f"Data:\n{df}")

    return df


def rename_human_readable(df):
    df = df.copy()
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

    df = df.rename(
        columns={
            "epsilon0": r"$\rho_0$",
            "epsilon1": r"$\rho$",
        }
    )

    df["stride_type"] = df["stride_type"].cat.remove_unused_categories()
    df["count_type"] = df["count_type"].cat.remove_unused_categories()

    logger.info(
        "stride_types remaining: " + str(list(df["stride_type"].cat.categories))
    )
    logger.info("count_types remaining: " + str(list(df["count_type"].cat.categories)))

    return df


class Plotter:
    def __init__(self, df: pd.DataFrame, outputname: str):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("Filtering data...")
        df = self.filter_data(df)

        self.logger.info("Make categories human readable...")
        df = rename_human_readable(df)

        self.logger.info("Plotting the graph...")
        self.plot(df)

        if outputname:
            self.logger.info("Writing the graph to its destination: " + outputname)
            plt_write(outputname)

    def plot(self, df: pd.DataFrame):
        """To be overridden"""
        pass

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """To be overridden"""
        pass


class BoxPlotSuccessChecksVsWeight(Plotter):
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        e = "True" if GRID_WEIGHTS else "epsilon1 == 1.0"
        return df.query(
            e + r" and weight % 10 == 0 and stride_type == 'checks' and count_type == 'remaining-flips' and success == True"
        )

    def plot(self, df: pd.DataFrame):
        alg = df["alg"].iloc[0]
        g = sns.catplot(
            data=df,
            x="stride",
            y="weight",
            col=r"$\rho$" if GRID_WEIGHTS else None,
            col_wrap=2 if GRID_WEIGHTS else None,
            orient="h",
            kind="box",
            dodge=True,
            # order=[1.0, 0.995, 0.95, 0.9],
            palette="cubehelix_r",
            linewidth=0.1,
            fliersize=1,
        )
        #g.set_titles("")
        g.set(xlim=(0, None))
        g.set_axis_labels("parity checks", "column weight")

class LinePlotChecksRemainingBitFlips(Plotter):
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        w = "weight % 10 == 0" if GRID_WEIGHTS else "weight == 50"
        return df.query(
            w + r" and stride_type == 'checks' and count_type == 'remaining-flips'"
        )

    def plot(self, df: pd.DataFrame):
        g = sns.relplot(
            data=df,
            x="stride",
            y="count",
            row="count_type",
            col="stride_type",
            hue=r"$\rho$",
            kind="line",
            palette="colorblind",
        )
        g.set_titles("")
        g.set_axis_labels("Parity checks", "Remaining bit-flips")


class BoxPlotSuccessOracleCalls(Plotter):
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        w = "weight % 10 == 0" if GRID_WEIGHTS else "weight == 50"
        return df.query(
            w + r" and stride_type == 'oracle_calls' and count_type == 'remaining-flips' and success == True"
        )

    def plot(self, df: pd.DataFrame):
        g = sns.catplot(
            data=df,
            x="stride",
            y=r"$\rho$",
            col="weight" if GRID_WEIGHTS else None,
            col_wrap=2 if GRID_WEIGHTS else None,
            orient="h",
            kind="box",
            # order=[1.0, 0.995, 0.95, 0.9],
            palette="colorblind",
            linewidth=0.1,
            fliersize=1,
        )
        #g.set_titles("")
        g.set(xlim=(0, None))
        g.set_axis_labels("Oracle calls", r"$\rho$")


class DescribeData(Plotter):
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.query(
            r"weight % 10 == 0 and stride_type == 'checks' and count_type == 'remaining-flips' and success == True"
        )

    def plot(self, df: pd.DataFrame):
        desc = df.groupby([r"$\rho$", "weight", "stride_type"])["stride"].describe()
        self.logger.info(f"Describe data: \n{desc}")


def view_hqc_simulation_csv(csv_file):

    df = load_data(csv_file)

    DescribeData(df, None)
    BoxPlotSuccessChecksVsWeight(df, "BoxPlotSuccessChecksVsWeight.pgf")
    #LinePlotChecksRemainingBitFlips(df, "LinePlotChecksRemainingBitFlips.pgf")
    BoxPlotSuccessOracleCalls(df, "BoxPlotSuccessOracleCalls.pgf")


def round_stride_of_type(df, stride_type, multiple_of):
    cond = df["stride_type"] == stride_type
    half = multiple_of // 2
    df.loc[cond, "stride"] = (
        df.loc[cond, "stride"].apply(lambda s: (s + half) / multiple_of).astype(int)
        * multiple_of
    )
    return df
