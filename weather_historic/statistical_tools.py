import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class StatisticalTools():

    def __init__(self, csv_path: str,):
        self.csv_path = csv_path

    def analyze_precip_extremes(
        self,
        date_col: str = "date",
        precip_col: str = "precip",
        lower_q: float = 5.0,
        upper_q: float = 95.0,
        make_plot: bool = True
    ):
        """
        Read multi-year precipitation time series, compute distribution,
        and identify events below lower_q and above upper_q percentiles.

        Parameters
        ----------
        csv_path : str
            Path to CSV file.
        date_col : str
            Name of the date column in the CSV.
        precip_col : str
            Name of the precipitation column in the CSV.
        lower_q : float
            Lower percentile (e.g. 5 for 5%).
        upper_q : float
            Upper percentile (e.g. 95 for 95%).
        make_plot : bool
            If True, plot empirical CDF and mark percentile thresholds.

        Returns
        -------
        df : pd.DataFrame
            Original dataframe with an extra column 'extreme_class' indicating:
            'low_extreme', 'high_extreme', or 'normal'.
        thresholds : dict
            Dictionary with 'lower' and 'upper' percentile thresholds.
        """

        # 1. Read and clean data
        df = pd.read_csv(self.csv_path)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        # Drop NaNs from precipitation
        valid = df[precip_col].notna()
        precip = df.loc[valid, precip_col].values

        if precip.size == 0:
            raise ValueError("No valid precipitation data found.")

        # 2. Compute percentile thresholds
        lower_thr = np.percentile(precip, lower_q)
        upper_thr = np.percentile(precip, upper_q)
        thresholds = {"lower": lower_thr, "upper": upper_thr}

        # 3. Classify events
        extreme_class = np.full(df.shape[0], "normal", dtype=object)
        extreme_class[(df[precip_col] <= lower_thr) & valid] = "low_extreme"
        extreme_class[(df[precip_col] >= upper_thr) & valid] = "high_extreme"
        df["extreme_class"] = extreme_class

        # 4. Optional: plot empirical CDF (distribution curve)
        if make_plot:
            sorted_vals = np.sort(precip)
            ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

            plt.figure()
            plt.plot(sorted_vals, ecdf)
            plt.axvline(lower_thr, linestyle="--", label=f"{lower_q}th pct = {lower_thr:.2f}")
            plt.axvline(upper_thr, linestyle="--", label=f"{upper_q}th pct = {upper_thr:.2f}")
            plt.xlabel("Total precipitation")
            plt.ylabel("Cumulative probability")
            plt.title("Empirical CDF of precipitation")
            plt.legend()
            plt.grid(True)
            plt.show()

        return df, thresholds
