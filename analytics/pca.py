# ============================================================================
# analytics/pca.py — Rolling PCA residual computation
# ============================================================================
# Computes rolling PCA on yield curve changes and returns per-tenor residuals.
# Residuals = actual yield change minus PCA-reconstructed yield change.
# These residuals are the core input to the Z-score signal generator.
#
# Plotting functions are intentionally excluded — kept in the notebook scratchpad
# so this module runs headlessly in run_daily.py without a display.
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import config


def compute_pca_residuals(
    rates_data: pd.DataFrame,
    window: int = config.PCA_WINDOW,
    n_components: int = config.N_COMPONENTS,
    mode: str = config.MODE,
) -> pd.DataFrame:
    """
    Compute rolling PCA residuals for each tenor.

    For each date T, fits PCA on yield changes over the rolling window
    ending at T-1 (strict look-ahead prevention), reconstructs yields
    using the top n_components PCs, and computes the residual as:
        residual[T] = actual_yield_change[T] - reconstructed_yield_change[T]

    Parameters
    ----------
    rates_data : pd.DataFrame
        Output of FetchRates(). Must contain a 'Date' column and all
        10 tenor columns defined in config.TENORS.
    window : int
        Rolling window size in trading days for PCA fitting.
        Default: config.PCA_WINDOW (20).
    n_components : int
        Number of principal components to retain.
        Default: config.N_COMPONENTS (2).
    mode : str
        Data mode — 'EOD', 'intraday', or '5day'. Label only; no branching.

    Returns
    -------
    pd.DataFrame
        Daily PCA residuals indexed by date.
        Columns: config.TENORS (10 tenor columns).
        First (window + 1) rows are NaN due to warmup.

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

    # Compute daily yield changes
    yield_changes = df[tenors].diff().dropna()

    # Output container
    residuals = pd.DataFrame(index=yield_changes.index, columns=tenors, dtype=float)

    print(f"\nRolling PCA explained variance (sampled annually):")
    print(f"  Window={window}d | n_components={n_components} | "
          f"Mode={mode}")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for i in range(window, len(yield_changes)):
            # Fit window: strictly excludes current date (look-ahead prevention)
            fit_window = yield_changes.iloc[i - window:i]
            current_change = yield_changes.iloc[i]
            current_date = yield_changes.index[i]

            try:
                # Scale
                scaler = StandardScaler()
                scaled_window = scaler.fit_transform(fit_window)

                # Fit PCA on window
                pca = PCA(n_components=n_components)
                pca.fit(scaled_window)

                # Log explained variance on first valid date and every 252 days
                # (approximately annually) so drift in factor structure is visible
                if i == window or (i - window) % 252 == 0:
                    ev = pca.explained_variance_ratio_
                    cumulative_ev = ev[:n_components].sum()
                    pc_str = '  '.join(
                        f'PC{j+1}={ev[j]*100:.1f}%'
                        for j in range(n_components)
                    )
                    print(f"  [{current_date.date()}] {pc_str}  "
                          f"Cumulative={cumulative_ev*100:.1f}%")

                # Project current day's change into PC space and reconstruct
                current_scaled = scaler.transform(current_change.values.reshape(1, -1))
                scores = pca.transform(current_scaled)
                reconstructed_scaled = pca.inverse_transform(scores)
                reconstructed = scaler.inverse_transform(reconstructed_scaled).flatten()

                # Residual = actual - reconstructed
                residuals.loc[current_date] = current_change.values - reconstructed

            except Exception:
                # Leave as NaN on computation failure
                residuals.loc[current_date] = np.nan

    return residuals.astype(float)
