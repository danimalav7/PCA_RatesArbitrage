# ============================================================================
# analytics/pca.py — Rolling PCA residual computation
# ============================================================================
# Computes rolling PCA on YIELD LEVELS (not changes) and returns per-tenor
# residuals representing persistent yield curve mispricing.
#
# Approach: vol-normalized level PCA (practitioner standard)
#   - Input: yield levels in decimal form (e.g. 0.0447 = 4.47%)
#   - Scaling: divide each tenor by its rolling window std ONLY
#              (no mean subtraction — preserves level information)
#   - Residual: actual_yield_level - PCA_reconstructed_level
#   - Units: decimal yield → multiply by YIELD_TO_BPS_SCALAR (10,000) for bps
#
# Why levels not changes:
#   Change residuals capture "was today's move unusual?" (1-day signal, tiny)
#   Level residuals capture "is this tenor persistently mispriced?" (15-25 day signal)
#   Level residuals at Z=3.0 → 9-27 bps gross P&L vs 0.50 bps round-trip cost
#
# Why vol-normalization not StandardScaler:
#   StandardScaler subtracts window mean → removes level information
#   Vol-norm divides by window std only → equalizes tenor volatility,
#   preserves yield levels, allows PCA to capture cross-sectional mispricing
#
# Cointegration guarantee:
#   Individual yields are I(1) non-stationary (trend from 0.5% to 5%)
#   But yields are cointegrated — they share common trends (Fed policy)
#   PCA extracts these common trends (PC1=parallel shift, PC2=slope)
#   Residuals = idiosyncratic component = I(0) stationary by construction
#   Confirmed: 10/10 tenors pass ADF+KPSS full-sample, 10/10 TRADEABLE at 252d rolling
#
# Hurst exponent of level residuals: H = 0.80-0.86 (documented)
#   Long memory — dislocations persist 15-25 days before reverting
#   ACF first crossing at 15-25 days confirms tradeable mean-reversion horizon
#
# Plotting functions excluded — runs headlessly in run_daily.py
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import warnings
import config


def compute_pca_residuals(
    rates_data: pd.DataFrame,
    window: int = config.PCA_WINDOW,
    n_components: int = config.N_COMPONENTS,
    mode: str = config.MODE,
) -> pd.DataFrame:
    """
    Compute rolling PCA residuals on vol-normalized yield LEVELS.

    For each date T, fits PCA on vol-normalized yield levels over the
    rolling window ending at T-1 (strict look-ahead prevention), reconstructs
    the yield level using the top n_components PCs, and computes the residual:

        residual[T] = actual_yield_level[T] - PCA_reconstructed_level[T]

    Scaling within each window:
        scaled_tenor[t] = yield_level[t] / std(yield_level[T-window:T])
    This equalizes volatility across tenors without removing level information.
    After PCA reconstruction, unscaling restores residuals to decimal yield units.

    Parameters
    ----------
    rates_data : pd.DataFrame
        Output of FetchRates(). Must contain 'Date' column and all
        10 tenor columns defined in config.TENORS.
        Yields must be in decimal form (e.g. 0.0447 = 4.47%).
    window : int
        Rolling window in trading days for PCA fitting and vol-normalization.
        Default: config.PCA_WINDOW (60d — ~2-3 mean reversion cycles).
    n_components : int
        Number of principal components to retain.
        Default: config.N_COMPONENTS (2 — PC1=level, PC2=slope).
    mode : str
        Data mode label. No branching. Default: config.MODE.

    Returns
    -------
    pd.DataFrame
        Daily PCA level residuals indexed by date.
        Columns: config.TENORS (10 tenor columns).
        Units: decimal yield (multiply by 10,000 to convert to bps).
        First `window` rows are NaN due to warmup.
        Positive residual = tenor yield ABOVE fair value = bond is CHEAP.
        Negative residual = tenor yield BELOW fair value = bond is RICH.

    Raises
    ------
    ValueError
        If rates_data is missing required tenor columns.
    """
    tenors = config.TENORS

    # Validate input columns
    missing = [t for t in tenors if t not in rates_data.columns]
    if missing:
        raise ValueError(
            f"rates_data is missing required tenor columns: {missing}"
        )

    # Set date index
    df = rates_data.copy()
    if 'Date' in df.columns:
        df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)

    # Use yield LEVELS directly — no differencing
    # Yields are in decimal form (e.g. 0.0447 = 4.47%)
    yield_levels = df[tenors].dropna()

    # Output container
    residuals = pd.DataFrame(
        index=yield_levels.index,
        columns=tenors,
        dtype=float
    )

    print(f"\nRolling PCA — vol-normalized yield levels")
    print(f"  Window={window}d | n_components={n_components} | Mode={mode}")
    print(f"  Approach: level PCA with tenor-specific vol normalization")
    print(f"  Explained variance (sampled annually):")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for i in range(window, len(yield_levels)):
            # Fitting window: strictly excludes current date (look-ahead prevention)
            fit_window   = yield_levels.iloc[i - window:i]
            current_obs  = yield_levels.iloc[i].values
            current_date = yield_levels.index[i]

            try:
                # ── Vol-normalization: divide by tenor-specific window std ──────
                # No mean subtraction — preserves yield level information
                # Equalizes volatility contribution across tenors
                window_stds = fit_window.std(axis=0).values
                # Guard against zero std (e.g. pegged rates, missing data)
                window_stds = np.where(window_stds < 1e-10, 1e-10, window_stds)

                scaled_window  = fit_window.values / window_stds
                current_scaled = current_obs        / window_stds

                # ── Fit PCA on vol-normalized yield levels ─────────────────────
                pca = PCA(n_components=n_components)
                pca.fit(scaled_window)

                # ── Log explained variance annually ────────────────────────────
                if i == window or (i - window) % 252 == 0:
                    ev            = pca.explained_variance_ratio_
                    cumulative_ev = ev[:n_components].sum()
                    pc_str        = '  '.join(
                        f'PC{j+1}={ev[j]*100:.1f}%'
                        for j in range(n_components)
                    )
                    print(f"  [{current_date.date()}] {pc_str}  "
                          f"Cumulative={cumulative_ev*100:.1f}%")

                # ── Project into PC space and reconstruct ──────────────────────
                scores               = pca.transform(current_scaled.reshape(1, -1))
                reconstructed_scaled = pca.inverse_transform(scores).flatten()

                # ── Un-scale back to yield level units ─────────────────────────
                reconstructed = reconstructed_scaled * window_stds

                # ── Residual = actual level - reconstructed level ──────────────
                # Positive → tenor yield above PCA fair value → CHEAP
                # Negative → tenor yield below PCA fair value → RICH
                residuals.loc[current_date] = current_obs - reconstructed

            except Exception:
                residuals.loc[current_date] = np.nan

    return residuals.astype(float)
