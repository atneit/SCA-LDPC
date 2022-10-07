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
    w = 4.8
    h = w if GRID_WEIGHTS else 2
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

    to_convert = [
        "label",
        "alg",
        "stride_type",
        "count_type",
        "success",
        "epsilon0",
        "epsilon1",
    ]
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

    return df


def hqc_csv_rename_human_readable(df):
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

    df["epsilon1"] = df["epsilon1"].cat.rename_categories(
        {
            "0.9": r"$\mathcal{O}_{\mathrm{HQC}}^{0.9}$",
            "0.95": r"$\mathcal{O}_{\mathrm{HQC}}^{0.95}$",
            "0.995": r"$\mathcal{O}_{\mathrm{HQC}}^{0.995}$",
            "1.0": r"$\mathcal{O}_{\mathrm{HQC}}^{\mathrm{ideal}}$",
            "miss-use": r"$\mathcal{O}_{\mathrm{HQC}}^{1.0}$",
        }
    )

    df = df.rename(
        columns={
            "epsilon0": r"$\rho_1$",
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
        df = self.rename_human_readable(df)

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

    def rename_human_readable(self, df):
        """May be overriden"""
        return hqc_csv_rename_human_readable(df)


class BoxPlotSuccessChecksVsWeight(Plotter):
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        e = "True" if GRID_WEIGHTS else "epsilon1 == 'miss-use'"
        return df.query(
            e
            + r" and weight % 10 == 0 and stride_type == 'checks' and count_type == 'remaining-flips' and success == True"
        )

    def plot(self, df: pd.DataFrame):
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
        # g.set_titles("")
        # g.set(xlim=(0, None))
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
            w
            + r" and stride_type == 'oracle_calls' and count_type == 'remaining-flips' and success == True"
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
        # g.set_titles("")
        # g.set(xlim=(0, None))
        g.set_axis_labels("Oracle calls", "")


class DescribeData(Plotter):
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.query(
            r"weight == 50 and stride_type == 'oracle_calls' and count_type == 'remaining-flips' and success == True"
        )

    def plot(self, df: pd.DataFrame):
        self.logger.info(df)
        desc = df.groupby([r"$\rho$", "weight", "stride_type"])["stride"].describe()
        self.logger.info(f"Describe data: \n{desc}")


def view_hqc_simulation_csv(csv_file):

    df = load_data(csv_file)

    DescribeData(df, None)
    BoxPlotSuccessChecksVsWeight(df, "BoxPlotSuccessChecksVsWeight.pgf")
    # LinePlotChecksRemainingBitFlips(df, "LinePlotChecksRemainingBitFlips.pgf")
    BoxPlotSuccessOracleCalls(df, "BoxPlotSuccessOracleCalls.pgf")


def round_stride_of_type(df, stride_type, multiple_of):
    cond = df["stride_type"] == stride_type
    half = multiple_of // 2
    df.loc[cond, "stride"] = (
        df.loc[cond, "stride"].apply(lambda s: (s + half) / multiple_of).astype(int)
        * multiple_of
    )
    return df


class OracleAccuracyPlotter(Plotter):
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["Measurements"] >= 64]

    def plot(self, df: pd.DataFrame):
        g = sns.lineplot(
            data=df, x="Measurements", y="Accuracy", palette="colorblind"
        )
        g.axes.set_xscale("log", base=2)
        g.set(ylim=(0.9, 1.0))
        
        # ax2 = g.axes.twinx() # create a second y axis
        # ax2.set_yticks([0.90, 0.95])
        # ax2.set_yticklabels([r"$\mathcal{O}_{\mathrm{HQC}}^{0.9}$", r"$\mathcal{O}_{\mathrm{HQC}}^{0.95}$"])
        # ax2.set(ylim=(0.8, 0.95))

    def rename_human_readable(self, df):
        """NOOP"""
        return df


def view_hqc_oracle_accuracy():

    # update 2022-10-06 with HP EliteBook 820-G4 notebook 
    # with Intel Core i5-7200@2.50GHz and 8Gb RAM running 
    # on Ubuntu 20.04 LTS using 2^18 profiling steps 
    # and 1000 trials. Code version was git commit id e49a035
    # acc = [
    #     0.0,
    #     0.546,
    #     0.713,
    #     0.915,
    #     0.903,
    #     0.138,
    #     0.911,
    #     0.957,
    #     0.946,
    #     0.947,
    #     0.966,
    #     0.947,
    #     0.948,
    #     0.952,
    #     0.962,
    #     0.953,
    #     0.969,
    #     0.969
    # ]

    # update 2022-10-07 with HP EliteBook 820-G4 notebook 
    # with Intel Core i5-7200@2.50GHz and 8Gb RAM running 
    # on Ubuntu 20.04 LTS using 2^18 profiling steps 
    # and 1000 trials. Code version was git commit id 00c3c65
    acc = [
        0.0,
        0.75,
        0.936,
        0.951,
        0.973,
        0.979,
        0.972,
        0.977,
        0.98,
        0.987,
        0.992,
        0.996,
        0.992,
        0.995,
        0.99,
        0.993,
        0.989
    ]
    N = len(acc)

    df = pd.DataFrame(
        {
            "Measurements": [2**x for x in range(N)],
            "Accuracy":  acc,
            "Legend": ["experiment"] * N 
        }
    )

    OracleAccuracyPlotter(df, "OracleAccuracy.pgf")
