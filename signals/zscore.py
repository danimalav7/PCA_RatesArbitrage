# ============================================================================
# signals/zscore.py — Z-score signal generation and signal card
# ============================================================================
# Provides:
#   compute_zscore()         — rolling Z-score normalization of residuals
#   generate_signal_card()   — full signal card dict for one tenor/date
#   scan_signals()           — scan all tenors on a given date
#
# Design principle: all DataFrames passed explicitly — no global closures.
# Every function takes residuals_df, z_score_df, rolling_adfs, etc. as
# parameters so this module runs correctly in both notebook and production.
#
# Entry threshold : config.Z_ENTRY_THRESHOLD (3.0)
# Exit threshold  : config.Z_EXIT_THRESHOLD  (1.0) — used by backtest only
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import config
from analytics.stationarity import get_stationarity_vote


def compute_zscore(
    residuals_df: pd.DataFrame,
    window: int = config.ZSCORE_WINDOW,
    mode: str = config.MODE,
) -> pd.DataFrame:
    """
    Compute rolling Z-scores for each tenor's residual series.

    Z-score[T] = (residual[T] - mean(residuals[T-window:T])) /
                  std(residuals[T-window:T])

    The rolling window strictly excludes date T (min_periods=window ensures
    no partial windows). First `window` rows are NaN.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Daily PCA residuals. Output of compute_pca_residuals().
    window : int
        Rolling normalization window in trading days.
        Default: config.ZSCORE_WINDOW (60).
    mode : str
        Data mode label. No branching.

    Returns
    -------
    pd.DataFrame
        Rolling Z-scores indexed by date, columns = config.TENORS.
        First `window` rows are NaN.
    """
    tenors = config.TENORS
    z_score_df = pd.DataFrame(index=residuals_df.index)

    for tenor in tenors:
        series = residuals_df[tenor]
        rolling_mean = series.rolling(
            window=window, min_periods=window
        ).mean()
        rolling_std = series.rolling(
            window=window, min_periods=window
        ).std()
        z_score_df[tenor] = (series - rolling_mean) / rolling_std

    print(f"Z-scores computed: {len(z_score_df)} trading days")
    print(f"Date range: {z_score_df.index[0].date()} → "
          f"{z_score_df.index[-1].date()}")
    print(f"Window: {window} days | Entry threshold: "
          f"±{config.Z_ENTRY_THRESHOLD}")

    return z_score_df.astype(float)


def generate_signal_card(
    date: pd.Timestamp,
    tenor: str,
    residuals_df: pd.DataFrame,
    z_score_df: pd.DataFrame,
    rolling_adfs: dict,
    rolling_adf_60d: pd.DataFrame,
    rolling_kpss_60d: pd.DataFrame,
    acf_summary_df: pd.DataFrame,
    auction_calendar: pd.DataFrame,
    segment_regime_df: pd.DataFrame,
    get_auction_flag_fn: callable,
    mode: str = config.MODE,
) -> dict:
    """
    Generate a complete signal card for a given tenor on a given date.

    Parameters
    ----------
    date : pd.Timestamp
        Analysis date.
    tenor : str
        Tenor to evaluate (e.g. '5Yr').
    residuals_df : pd.DataFrame
        Daily PCA residuals.
    z_score_df : pd.DataFrame
        Rolling Z-scores. Output of compute_zscore().
    rolling_adfs : dict
        All rolling ADF windows. Output of compute_rolling_adf().
        Keys are window sizes (int).
    rolling_adf_60d : pd.DataFrame
        60-day rolling ADF p-values (rolling_adfs[60]).
        Passed separately for convenience since it's used for entry gate.
    acf_summary_df : pd.DataFrame
        ACF/PACF summary. Output of compute_acf_summary().
    auction_calendar : pd.DataFrame
        Auction calendar DataFrame.
    segment_regime_df : pd.DataFrame
        Segment regime flags (dead variable — display only, not eligibility).
    get_auction_flag_fn : callable
        Function with signature:
        get_auction_flag_fn(date, tenor, auction_calendar, mode) ->
        (flag: str, info: dict)
    mode : str
        Data mode. Default: config.MODE.

    Returns
    -------
    dict
        Complete signal card with all context fields for trading decisions.
    """
    date = pd.Timestamp(date)
    card = {
        'date':  date.date(),
        'tenor': tenor,
        'mode':  mode,
    }

    # ── 1. Core signal metrics ────────────────────────────────────────────────
    daily_residual = (
        float(residuals_df.loc[date, tenor])
        if date in residuals_df.index and tenor in residuals_df.columns
        else np.nan
    )
    card['daily_residual_bps'] = daily_residual

    z_score = (
        float(z_score_df.loc[date, tenor])
        if date in z_score_df.index and tenor in z_score_df.columns
        else np.nan
    )
    card['z_score'] = z_score

    # Signal direction
    # Z > +threshold: yield ABOVE fair value → CHEAP → LONG (yield falls)
    # Z < -threshold: yield BELOW fair value → RICH  → SHORT (yield rises)
    threshold = config.Z_ENTRY_THRESHOLD
    if np.isnan(z_score):
        signal_direction = 'FLAT'
        bond_status = None
        expected_move = None
    elif z_score > threshold:
        signal_direction = 'LONG'
        bond_status = 'CHEAP'
        expected_move = 'YIELD_FALLS'
    elif z_score < -threshold:
        signal_direction = 'SHORT'
        bond_status = 'RICH'
        expected_move = 'YIELD_RISES'
    else:
        signal_direction = 'FLAT'
        bond_status = None
        expected_move = None

    card['signal_direction'] = signal_direction
    card['bond_status']      = bond_status
    card['expected_move']    = expected_move

    # ── 2. Stationarity regime ────────────────────────────────────────────────
    # Entry gate: dual ADF + KPSS (both 60d rolling)
    rolling_adf_pval = (
        float(rolling_adf_60d.loc[date, tenor])
        if date in rolling_adf_60d.index and tenor in rolling_adf_60d.columns
        else np.nan
    )
    card['rolling_adf_pvalue'] = rolling_adf_pval

    rolling_kpss_pval = (
        float(rolling_kpss_60d.loc[date, tenor])
        if date in rolling_kpss_60d.index and tenor in rolling_kpss_60d.columns
        else np.nan
    )
    card['rolling_kpss_pvalue'] = rolling_kpss_pval

    adf_stationary  = not np.isnan(rolling_adf_pval)  and rolling_adf_pval  < config.ADF_THRESHOLD
    kpss_stationary = not np.isnan(rolling_kpss_pval) and rolling_kpss_pval > config.KPSS_ENTRY_THRESHOLD

    if adf_stationary and kpss_stationary:
        tenor_stationarity = 'STATIONARY'
    elif not adf_stationary and not kpss_stationary:
        tenor_stationarity = 'NON-STATIONARY'
    elif np.isnan(rolling_adf_pval) or np.isnan(rolling_kpss_pval):
        tenor_stationarity = 'UNKNOWN'
    else:
        tenor_stationarity = 'AMBIGUOUS'
    card['tenor_stationarity_status'] = tenor_stationarity

    # Multi-window vote for escalation display
    vote_count = get_stationarity_vote(date, tenor, rolling_adfs)
    card['stationarity_vote_count'] = (
        vote_count if vote_count is not None else np.nan
    )

    if vote_count is None or vote_count == 4:
        escalation = 'CLEAR'
    elif vote_count == 3:
        escalation = 'MONITOR'
    elif vote_count == 2:
        escalation = 'WARNING'
    elif vote_count == 1:
        escalation = 'DETERIORATING'
    else:  # vote_count == 0
        escalation = 'EXIT TRIGGER'
    card['stationarity_escalation'] = escalation

    # ── 3. Segment context ────────────────────────────────────────────────────
    segment_name = config.SEGMENT_MAP.get(tenor, 'unknown')
    card['curve_segment'] = segment_name

    # Segment regime status — display only, not used for eligibility
    segment_blocked = False  # segment suppression removed from live logic
    card['segment_regime_status'] = 'SUPPRESSED' if segment_blocked else 'ACTIVE'

    # Non-stationary tenors in segment (60d based, informational)
    seg_tenors = config.SEGMENT_TENORS.get(segment_name, [])
    non_stationary_in_segment = [
        t for t in seg_tenors
        if date in rolling_adf_60d.index
        and t in rolling_adf_60d.columns
        and rolling_adf_60d.loc[date, t] > config.ADF_THRESHOLD
    ]
    card['non_stationary_tenors_in_segment'] = non_stationary_in_segment
    card['n_non_stationary_in_segment']      = len(non_stationary_in_segment)

    # ── 4. Auction context ────────────────────────────────────────────────────
    auction_flag, auction_info = get_auction_flag_fn(
        date, tenor, auction_calendar, mode=mode
    )
    card['auction_flag']        = auction_flag
    card['next_auction_date']   = auction_info['next_auction_date']
    card['days_to_next_auction'] = auction_info['days_to_next']

    # ── 5. Mean reversion context ─────────────────────────────────────────────
    if tenor in acf_summary_df.index:
        first_cross = acf_summary_df.loc[tenor, 'First ACF Cross']
        acf_horizon = np.nan if first_cross == 'Never' else int(first_cross)
    else:
        acf_horizon = np.nan

    card['acf_mean_reversion_horizon_days'] = acf_horizon
    card['expected_hold_period_days'] = (
        f"{max(1, int(acf_horizon) - 2)}-{int(acf_horizon) + 2}"
        if not np.isnan(acf_horizon) else 'N/A'
    )

    # ── 6. PC3 variance overlay ───────────────────────────────────────────────
    pc3_var = np.nan
    if date in residuals_df.index:
        date_idx = residuals_df.index.get_loc(date)
        pca_window = config.PCA_WINDOW
        if date_idx >= pca_window:
            window_data = residuals_df.iloc[date_idx - pca_window:date_idx]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    scaler  = StandardScaler()
                    scaled  = scaler.fit_transform(
                        window_data[config.TENORS]
                    )
                    pca_3   = PCA(n_components=3)
                    pca_3.fit(scaled)
                    pc3_var = (
                        pca_3.explained_variance_ratio_[2]
                        if pca_3.n_components_ >= 3 else np.nan
                    )
            except Exception:
                pc3_var = np.nan

    card['pc3_explained_variance'] = (
        float(pc3_var) if not np.isnan(pc3_var) else np.nan
    )
    pc3_elevated = not np.isnan(pc3_var) and pc3_var > config.PC3_ELEVATED_THRESHOLD
    card['pc3_elevated_flag'] = pc3_elevated

    # ── 7. Overall trade eligibility ──────────────────────────────────────────
    blocked_reasons = []

    if auction_flag == 'SUPPRESS':
        blocked_reasons.append('AUCTION-SUPPRESS')
    if tenor_stationarity in ('NON-STATIONARY', 'AMBIGUOUS', 'UNKNOWN'):
        blocked_reasons.append(f'STATIONARITY-{tenor_stationarity}')
    if pc3_elevated:
        blocked_reasons.append('PC3-ELEVATED')
    if signal_direction == 'FLAT':
        blocked_reasons.append('FLAT-SIGNAL')

    if blocked_reasons:
        card['trade_eligibility'] = (
            f"BLOCKED ({', '.join(blocked_reasons)})"
        )
    elif vote_count is not None and vote_count == 0:
        card['trade_eligibility'] = (
            'BLOCKED — Non-Stationary Regime (4/4 windows flagging)'
        )
    elif vote_count is not None and vote_count == 2:
        card['trade_eligibility'] = (
            'WARNING — Regime Deterioration Developing '
            '(2/4 windows non-stationary)'
        )
    else:
        card['trade_eligibility'] = 'ELIGIBLE'

    return card


def scan_signals(
    date: pd.Timestamp,
    residuals_df: pd.DataFrame,
    z_score_df: pd.DataFrame,
    rolling_adfs: dict,
    rolling_adf_60d: pd.DataFrame,
    rolling_kpss_60d: pd.DataFrame,
    acf_summary_df: pd.DataFrame,
    auction_calendar: pd.DataFrame,
    segment_regime_df: pd.DataFrame,
    get_auction_flag_fn: callable,
    mode: str = config.MODE,
) -> pd.DataFrame:
    """
    Scan all tenors on a given date and return signal cards as a DataFrame.

    Parameters
    ----------
    date : pd.Timestamp
        Date to scan.
    All other parameters are passed through to generate_signal_card().
    mode : str
        Data mode. Default: config.MODE.

    Returns
    -------
    pd.DataFrame
        One row per tenor, sorted by absolute Z-score descending.
    """
    cards = []
    for tenor in config.TENORS:
        card = generate_signal_card(
            date=date,
            tenor=tenor,
            residuals_df=residuals_df,
            z_score_df=z_score_df,
            rolling_adfs=rolling_adfs,
            rolling_adf_60d=rolling_adf_60d,
            rolling_kpss_60d=rolling_kpss_60d,
            acf_summary_df=acf_summary_df,
            auction_calendar=auction_calendar,
            segment_regime_df=segment_regime_df,
            get_auction_flag_fn=get_auction_flag_fn,
            mode=mode,
        )
        cards.append(card)

    df = pd.DataFrame(cards)
    df['abs_z_score'] = df['z_score'].abs()
    df = df.sort_values(
        'abs_z_score', ascending=False, na_position='last'
    ).reset_index(drop=True)
    df = df.drop(columns=['abs_z_score'])
    return df
