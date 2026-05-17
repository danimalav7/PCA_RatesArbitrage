# ============================================================================
# analytics/stationarity.py — Stationarity testing and voting system
# ============================================================================
# Provides:
#   compute_rolling_adf()        — rolling ADF p-values across multiple windows
#   compute_rolling_kpss()       — rolling KPSS p-values for 60d entry gate
#   get_stationarity_vote()      — 4-window majority vote (returns raw int 0-4)
#   compute_voting_stationary_df() — full date×tenor boolean DataFrame
#   compute_acf_summary()        — ACF/PACF mean reversion horizon per tenor
#   compute_segment_regime_df()  — 60d-only segment suppression (dead variable,
#                                  kept for reference; not used in live logic)
#
# Entry gate  : rolling_adf_60d p-value < 0.05 (single window, most reliable)
# Exit gate   : get_stationarity_vote() == 0 (all 4 windows flagging)
# Escalation  : vote count drives CLEAR / MONITOR / WARNING / DETERIORATING /
#               EXIT TRIGGER display states
# ============================================================================

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
import config


def compute_rolling_adf(
    residuals_df: pd.DataFrame,
    windows: list = config.ROLLING_ADF_WINDOWS,
    mode: str = config.MODE,
) -> dict:
    """
    Compute rolling ADF p-values across multiple window sizes.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Daily PCA residuals. Output of compute_pca_residuals().
    windows : list of int
        Rolling window sizes in trading days.
        Default: config.ROLLING_ADF_WINDOWS ([10, 15, 20, 60]).
    mode : str
        Data mode label. No branching.

    Returns
    -------
    dict
        Keys are window sizes (int). Values are pd.DataFrames of ADF p-values,
        indexed by date, columns = config.TENORS.
        Example: {10: df_10d, 15: df_15d, 20: df_20d, 60: df_60d}
    """
    tenors = config.TENORS
    result = {}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for window in windows:
            pvals_by_tenor = {}

            for tenor in tenors:
                series = residuals_df[tenor].dropna()
                pvals = []
                roll_dates = []

                for i in range(window, len(series) + 1):
                    window_series = series.iloc[i - window:i]
                    try:
                        result_adf = adfuller(window_series, autolag='AIC')
                        pvals.append(result_adf[1])
                    except Exception:
                        pvals.append(np.nan)
                    roll_dates.append(series.index[i - 1])

                pvals_by_tenor[tenor] = pd.Series(pvals, index=roll_dates)

            result[window] = pd.DataFrame(pvals_by_tenor)

    print(f"Rolling ADF computed — windows: {windows}")
    for w in windows:
        df = result[w]
        print(f"  {w:>3}d: {len(df)} dates  "
              f"({df.index[0].date()} → {df.index[-1].date()})")

    return result


def compute_rolling_kpss(
    residuals_df: pd.DataFrame,
    window: int = config.ADF_ENTRY_WINDOW,
    mode: str = config.MODE,
) -> pd.DataFrame:
    """
    Compute rolling KPSS p-values using a single window size.

    KPSS: H0 = stationary. p > config.KPSS_ENTRY_THRESHOLD → fail to reject
    → stationary evidence. Used alongside rolling ADF as dual entry gate.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Daily PCA residuals. Output of compute_pca_residuals().
    window : int
        Rolling window size in trading days.
        Default: config.ADF_ENTRY_WINDOW (60).
    mode : str
        Data mode label. No branching.

    Returns
    -------
    pd.DataFrame
        Rolling KPSS p-values indexed by date, columns = config.TENORS.
        Note: KPSS p-values are bounded [0.01, 0.10] by statsmodels
        interpolation table — values outside this range are clipped.
    """
    tenors = config.TENORS
    pvals_by_tenor = {}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for tenor in tenors:
            series = residuals_df[tenor].dropna()
            pvals      = []
            roll_dates = []

            for i in range(window, len(series) + 1):
                window_series = series.iloc[i - window:i]
                try:
                    result = kpss(
                        window_series,
                        regression='c',
                        nlags='auto'
                    )
                    pvals.append(result[1])
                except Exception:
                    pvals.append(np.nan)
                roll_dates.append(series.index[i - 1])

            pvals_by_tenor[tenor] = pd.Series(pvals, index=roll_dates)

    rolling_kpss_60d = pd.DataFrame(pvals_by_tenor)

    print(f"Rolling KPSS {window}d computed: {len(rolling_kpss_60d)} dates")
    print(f"Date range: {rolling_kpss_60d.index[0].date()} → "
          f"{rolling_kpss_60d.index[-1].date()}")

    return rolling_kpss_60d


def get_stationarity_vote(
    date: pd.Timestamp,
    tenor: str,
    rolling_adfs: dict,
    threshold: float = config.ADF_THRESHOLD,
    windows: list = None,
) -> int | None:
    """
    Vote across a specified subset of rolling ADF windows for a given date and tenor.

    Returns the raw count of windows voting stationary (p < threshold).
    Callers implement their own threshold logic:
        entry gate : rolling_adfs[60].loc[date, tenor] < threshold (ADF)
                     rolling_kpss_60d.loc[date, tenor] > threshold (KPSS)
        exit gate  : get_stationarity_vote(..., windows=config.EXIT_ADF_WINDOWS) == 0
                     Only 60d window used for exit — short windows lack statistical power
        escalation : uses all ROLLING_ADF_WINDOWS for display only (not for exit decisions)

    Parameters
    ----------
    date : pd.Timestamp
        Date to evaluate.
    tenor : str
        Tenor to evaluate (e.g. '5Yr').
    rolling_adfs : dict
        Output of compute_rolling_adf(). Keys are window sizes (int).
    threshold : float
        ADF p-value threshold for stationarity. Default: config.ADF_THRESHOLD (0.05).

    Returns
    -------
    int or None
        Raw count of windows voting stationary (0–4).
        None if fewer than 3 windows have data for this date — conservative default.
    """
    votes_stationary = 0
    available_windows = 0

    # Filter to specified windows only — if None, use all windows
    adfs_to_check = (
        {w: rolling_adfs[w] for w in windows if w in rolling_adfs}
        if windows is not None else rolling_adfs
    )

    for window_size, df_w in adfs_to_check.items():
        if date in df_w.index and tenor in df_w.columns:
            val = df_w.loc[date, tenor]
            if not np.isnan(val):
                available_windows += 1
                if val < threshold:
                    votes_stationary += 1

    min_windows = max(1, len(adfs_to_check) - 1)  # require at least n-1 windows to have data
    if available_windows < min_windows:
        return None  # insufficient data — conservative default

    return votes_stationary  # raw count (0-4)


def compute_voting_stationary_df(
    rolling_adfs: dict,
    tenors: list = config.TENORS,
) -> pd.DataFrame:
    """
    Build a full date × tenor boolean DataFrame from the stationarity vote.

    True  = vote_count >= 3 (3+ windows agree stationary)
    False = vote_count <  3 or None (conservative default)

    Parameters
    ----------
    rolling_adfs : dict
        Output of compute_rolling_adf().
    tenors : list
        Tenor list. Default: config.TENORS.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame indexed by date, columns = tenors.
    """
    # Union of all dates across all windows
    all_dates = pd.DatetimeIndex([])
    for df_w in rolling_adfs.values():
        all_dates = all_dates.union(df_w.index)

    voting_df = pd.DataFrame(False, index=all_dates, columns=tenors)

    for date in all_dates:
        for tenor in tenors:
            result = get_stationarity_vote(date, tenor, rolling_adfs)
            voting_df.loc[date, tenor] = False if result is None else (result >= 3)

    print(f"voting_stationary_df: {len(voting_df)} dates × {len(tenors)} tenors")
    return voting_df


def compute_acf_summary(
    residuals_df: pd.DataFrame,
    max_lags: int = 30,
    tenors: list = config.TENORS,
) -> pd.DataFrame:
    """
    Compute ACF/PACF mean reversion horizon summary per tenor.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Daily PCA residuals. Output of compute_pca_residuals().
    max_lags : int
        Maximum lags for ACF/PACF computation. Default: 30.
    tenors : list
        Tenor list. Default: config.TENORS.

    Returns
    -------
    pd.DataFrame
        Indexed by tenor. Columns:
        First ACF Cross, Decay Type, Sig. PACF Lags, Classification
    """
    def _first_acf_crossing(acf_vals, ci):
        return next(
            (lag for lag in range(1, len(acf_vals)) if abs(acf_vals[lag]) < ci),
            None
        )

    def _decay_type(acf_vals, first_crossing, ci):
        if first_crossing is None:
            return 'Never Crosses CI'
        if first_crossing <= 4 and (
            first_crossing == 1 or abs(acf_vals[first_crossing - 1]) > ci
        ):
            return 'Sharp Cutoff'
        return 'Gradual Decay'

    def _classify_acf(first_crossing):
        if first_crossing is None:
            return 'Unit Root — Do Not Trade'
        if first_crossing <= 5:
            return 'Fast (lags 1-5) — High Signal Quality'
        if first_crossing <= 10:
            return 'Moderate (lags 6-10) — Tradeable'
        return 'Slow (lag >10) — Caution'

    rows = []
    for tenor in tenors:
        series = residuals_df[tenor].dropna()
        ci = 1.96 / np.sqrt(len(series))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            acf_vals  = acf(series,  nlags=max_lags, fft=True)
            pacf_vals = pacf(series, nlags=max_lags, method='ywa')

        first_crossing = _first_acf_crossing(acf_vals, ci)
        sig_pacf_lags  = [
            lag for lag in range(1, max_lags + 1)
            if abs(pacf_vals[lag]) > ci
        ]

        rows.append({
            'Tenor':           tenor,
            'First ACF Cross': first_crossing if first_crossing is not None else 'Never',
            'Decay Type':      _decay_type(acf_vals, first_crossing, ci),
            'Sig. PACF Lags':  sig_pacf_lags if sig_pacf_lags else ['None'],
            'Classification':  _classify_acf(first_crossing),
        })

    return pd.DataFrame(rows).set_index('Tenor')


def compute_segment_regime_df(
    rolling_adf_60d: pd.DataFrame,
    mode: str = config.MODE,
) -> pd.DataFrame:
    """
    Compute segment-level regime suppression flags using 60d ADF only.

    NOTE: This is a dead variable kept for reference. Segment suppression
    was removed from live entry/exit logic — each tenor is gated individually
    by its own 60d ADF p-value. Do not use this in signal generation or
    backtest entry/exit conditions.

    Logic: segment is suppressed if 2+ of its tenors have ADF p > 0.05.
    Long End (30Yr only) can never reach the 2+ threshold — always False.

    Parameters
    ----------
    rolling_adf_60d : pd.DataFrame
        60-day rolling ADF p-values. rolling_adfs[60] from compute_rolling_adf().
    mode : str
        Data mode label. No branching.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame indexed by date, columns = segment names.
        True = segment suppressed. Not used in live logic.
    """
    segments = config.SEGMENT_TENORS
    regime = {}

    for seg_name, seg_tenors in segments.items():
        n_nonstationary = (rolling_adf_60d[seg_tenors] > 0.05).sum(axis=1)
        regime[seg_name] = n_nonstationary >= 2

    return pd.DataFrame(regime, index=rolling_adf_60d.index)


def compute_full_sample_stationarity(
    residuals_df: pd.DataFrame,
    tenors: list = config.TENORS,
) -> pd.DataFrame:
    """
    Full-sample ADF + KPSS stationarity tests for baseline eligibility.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Daily PCA residuals.
    tenors : list
        Tenor list. Default: config.TENORS.

    Returns
    -------
    pd.DataFrame
        Indexed by tenor. Columns: ADF Stat, ADF p-val, KPSS Stat,
        KPSS p-val, Conclusion (Stationary / Non-Stationary / Ambiguous).
    """
    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for tenor in tenors:
            series = residuals_df[tenor].dropna()

            adf_result  = adfuller(series, autolag='AIC')
            kpss_result = kpss(series, regression='c', nlags='auto')

            adf_stat, adf_pval   = adf_result[0],  adf_result[1]
            kpss_stat, kpss_pval = kpss_result[0], kpss_result[1]

            adf_stationary  = adf_pval  < 0.05
            kpss_stationary = kpss_pval > 0.05

            if adf_stationary and kpss_stationary:
                conclusion = 'Stationary'
            elif not adf_stationary and not kpss_stationary:
                conclusion = 'Non-Stationary'
            else:
                conclusion = 'Ambiguous'

            rows.append({
                'Tenor':      tenor,
                'ADF Stat':   round(adf_stat,  4),
                'ADF p-val':  '< 0.0001' if adf_pval < 0.0001 else f'{adf_pval:.4f}',
                'KPSS Stat':  round(kpss_stat, 4),
                'KPSS p-val': round(kpss_pval, 4),
                'Conclusion': conclusion,
            })

    return pd.DataFrame(rows).set_index('Tenor')
