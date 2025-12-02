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

    def analyze_cdd_distribution(
        precip: pd.Series,
        dry_threshold: float = 1.0,
        plot: bool = True
    ):
        """
        Compute the distribution and inverse cumulative distribution of 
        Consecutive Dry Days (CDD) from a daily precipitation time series.

        Parameters
        ----------
        precip : pd.Series
            Daily precipitation values in chronological order.
        dry_threshold : float
            Maximum precipitation (mm) to classify a day as "dry".
        plot : bool
            If True, plot CDF and inverse CDF.

        Returns
        -------
        cdd_lengths : np.ndarray
            Array of all dry spell lengths.
        cdf : pd.DataFrame
            DataFrame with columns ["CDD_length", "CDF"].
        inv_cdf : pd.DataFrame
            DataFrame with columns ["CDD_length", "ExceedanceProb"].
        """

        # 1. Identify dry days
        dry = precip.values <= dry_threshold

        # 2. Extract lengths of consecutive dry spells
        cdd_lengths = []
        current_length = 0

        for is_dry in dry:
            if is_dry:
                current_length += 1
            else:
                if current_length > 0:
                    cdd_lengths.append(current_length)
                current_length = 0

        # Catch trailing dry spell
        if current_length > 0:
            cdd_lengths.append(current_length)

        cdd_lengths = np.array(cdd_lengths)
        if len(cdd_lengths) == 0:
            raise ValueError("No dry spells found with this threshold.")

        # 3. Sort and compute empirical CDF
        sorted_lengths = np.sort(cdd_lengths)
        cdf_vals = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

        cdf = pd.DataFrame({
            "CDD_length": sorted_lengths,
            "CDF": cdf_vals
        })

        # 4. Compute inverse CDF (exceedance probability)
        exceedance = 1 - cdf_vals
        inv_cdf = pd.DataFrame({
            "CDD_length": sorted_lengths,
            "ExceedanceProb": exceedance
        })

        # 5. Optional: plotting
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # CDF plot
            ax[0].plot(sorted_lengths, cdf_vals)
            ax[0].set_xlabel("CDD length (days)")
            ax[0].set_ylabel("CDF")
            ax[0].set_title("Cumulative Distribution of CDD")
            ax[0].grid(True)

            # Inverse CDF (Exceedance) plot
            ax[1].plot(sorted_lengths, exceedance)
            ax[1].set_xlabel("CDD length (days)")
            ax[1].set_ylabel("P(CDD > x)")
            ax[1].set_title("Inverse CDF (Exceedance Probability)")
            ax[1].grid(True)

            plt.tight_layout()
            plt.show()

        return cdd_lengths, cdf, inv_cdf