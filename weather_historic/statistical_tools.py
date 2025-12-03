import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from eto_fao56 import eto_fao56

class StatisticalTools():

    def __init__(self, data_frame: pd.DataFrame = None, csv_path: str = None, date_col: str = "date", date_format: str = "%Y-%m-%d", precip_col: str = "precip"):
        # Read and clean data
        df = pd.read_csv(csv_path) if csv_path else data_frame
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        df.index = df[date_col]
        df = df.drop(columns=[date_col])
        self.df = df.sort_values(date_col)
        self.precip_col = precip_col

    def analyze_precip_extremes(
        self,
        lower_q: float = 5.0,
        upper_q: float = 95.0,
        make_plot: bool = True
    ):
        """
        Read multi-year precipitation time series, compute distribution,
        and identify events below lower_q and above upper_q percentiles.

        Parameters
        ----------
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

        # 1. Drop NaNs from precipitation
        valid = self.df[self.precip_col].notna()
        precip = self.df.loc[valid, self.precip_col].values

        if precip.size == 0:
            raise ValueError("No valid precipitation data found.")

        # 2. Compute percentile thresholds
        lower_thr = np.percentile(precip, lower_q)
        upper_thr = np.percentile(precip, upper_q)
        thresholds = {"lower": lower_thr, "upper": upper_thr}

        # 3. Classify events
        extreme_class = np.full(self.df.shape[0], "normal", dtype=object)
        extreme_class[(self.df[self.precip_col] <= lower_thr) & valid] = "low_extreme"
        extreme_class[(self.df[self.precip_col] >= upper_thr) & valid] = "high_extreme"
        self.df["extreme_class"] = extreme_class

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

        return self.df, thresholds

    def analyze_extreme_drought(
        self,
        dry_threshold: float = 1.0,
        plot: bool = True
    ):
        """
        Compute the distribution and inverse cumulative distribution of 
        Consecutive Dry Days (CDD) from a daily precipitation time series.

        Parameters
        ----------
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
        precip = self.df[self.precip_col]
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
    
    def count_extreme_precipitation_events_by_year(
        self,
        threshold: float = 9.6
    ):
        """
        Count number of days per year with precipitation >= threshold.

        Parameters
        ----------
        threshold : float
            Precipitation depth defining an extreme event (e.g. 9.6 mm).

        Returns
        -------
        counts : pd.DataFrame
            DataFrame with columns:
            - 'year'
            - 'n_extreme_events'
        """

        df = self.df.copy()

        # Filter days that exceed (or equal) the threshold
        mask = df[self.precip_col] >= threshold

        # Group by year and count
        counts = (
            df.loc[mask]
            .groupby(df.loc[mask].index.year)
            .size()
            .rename("n_extreme_events")
            .rename_axis("Year")
        )

        return counts
    
    def compute_spi(
            self,
            scale: int = 3,
            calib_start: str | None = None,
            calib_end: str | None = None
            ) -> pd.Series:
        """
        Compute Standardized Precipitation Index (SPI) from a monthly
        precipitation series using a gamma distribution and zero-precip correction.

        Parameters
        ----------
        scale : int
            Accumulation scale in months (e.g. 1, 3, 6, 12).
        calib_start, calib_end : str or None
            Optional calibration period (e.g. '1971-01', '2000-12').
            If None, the full period is used.

        Returns
        -------
        spi : pd.Series
            SPI time series (same index as the accumulated series).
        """
        precip = self.df[self.precip_col].sort_index().resample('ME').sum()

        # Accumulate to desired time scale (e.g., 3-month totals)
        acc = precip.rolling(window=scale, min_periods=scale).sum()
        
        # Calibration subset
        if calib_start is not None and calib_end is not None:
            calib = acc[calib_start:calib_end].dropna()
        else:
            calib = acc.dropna()

        if calib.empty:
            raise ValueError("Calibration period has no data.")

        # Separate zeros and non-zeros
        calib_nonzero = calib[calib > 0.0]
        q_zero = 1.0 - len(calib_nonzero) / len(calib)

        if len(calib_nonzero) < 10:
            raise ValueError("Too few non-zero values for reliable gamma fit.")

        # Fit gamma distribution (forcing location to 0)
        shape, loc, scale_param = stats.gamma.fit(calib_nonzero.values, floc=0.0)

        # Compute CDF for all values (accumulated)
        spi_values = []
        for x in acc:
            if np.isnan(x):
                spi_values.append(np.nan)
                continue

            if x <= 0:
                # Entirely within zero-mass probability
                H = q_zero
            else:
                G = stats.gamma.cdf(x, shape, loc=loc, scale=scale_param)
                H = q_zero + (1.0 - q_zero) * G

            # Avoid 0 or 1 exactly
            H = np.clip(H, 1e-6, 1 - 1e-6)

            # Transform to standard normal
            spi_values.append(stats.norm.ppf(H))

        spi = pd.Series(spi_values, index=acc.index, name=f"SPI-{scale}")
        return spi
    
    def compute_spei(
            self,
            pet: pd.Series,
            scale: int = 3,
            calib_start: str | None = None,
            calib_end: str | None = None
            ) -> pd.Series:
        """
        Compute Standardized Precipitation Evapotranspiration Index (SPEI)
        from monthly precipitation and PET series.

        Uses Pearson Type III distribution on the accumulated climatic
        water balance D = P - PET.

        Parameters
        ----------
        pet : pd.Series
            Monthly potential evapotranspiration [mm], same index as precip.
        scale : int
            Accumulation scale in months (e.g. 1, 3, 6, 12).
        calib_start, calib_end : str or None
            Optional calibration period (e.g. '1971-01', '2000-12').

        Returns
        -------
        spei : pd.Series
            SPEI time series (same index as the accumulated series).
        """
        # Align series
        precip, pet = self.precip.sort_index().align(pet.sort_index(), join="inner")

        # Climatic water balance
        D = precip - pet

        # Accumulate to the chosen time scale
        accD = D.rolling(window=scale, min_periods=scale).sum()

        # Calibration subset
        if calib_start is not None and calib_end is not None:
            calib = accD[calib_start:calib_end].dropna()
        else:
            calib = accD.dropna()

        if len(calib) < 30:
            raise ValueError("Too few calibration values for Pearson III fit.")

        # Fit Pearson type III (can handle negative values)
        skew, loc, scale_param = stats.pearson3.fit(calib.values)

        # Compute CDF and convert to standard normal
        spei_values = []
        for x in accD:
            if np.isnan(x):
                spei_values.append(np.nan)
                continue

            F = stats.pearson3.cdf(x, skew, loc=loc, scale=scale_param)
            F = np.clip(F, 1e-6, 1 - 1e-6)
            spei_values.append(stats.norm.ppf(F))

        spei = pd.Series(spei_values, index=accD.index, name=f"SPEI-{scale}")
        return spei