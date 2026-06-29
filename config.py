# ============================================================================
# config.py — Central parameter store for PCA Rates Arbitrage Strategy
# ============================================================================
# All strategy parameters live here. Import this module in every other module.
# To change a parameter, change it here only — never hardcode in other files.
# ============================================================================

# ── Data ──────────────────────────────────────────────────────────────────────
MODE        = 'EOD'           # 'EOD' | 'intraday' | '5day'
START_DATE  = '2005-12-01'   # ensures first valid Z-score by Mar 2006
END_DATE    = None            # None = today

# ── PCA ───────────────────────────────────────────────────────────────────────
# Approach: PCA on yield LEVELS with vol-normalization (practitioner standard)
#   - Yields are I(1) non-stationary individually but cointegrated cross-sectionally
#   - PCA extracts common trends (PC1=parallel shift, PC2=slope change)
#   - Residuals are I(0) stationary by construction — the tradeable signal
#   - Vol-normalization: divide each tenor by its rolling window std (no mean subtraction)
#     This equalizes volatility contribution across tenors while preserving yield levels
#     StandardScaler is NOT used — it would remove level information via mean subtraction
#
# Hurst exponent of level residuals: H = 0.80-0.86 (documented, not a parameter)
#   Interpretation: long memory — dislocations persist 15-25 days before reverting
#   This is the tradeable signal duration, NOT a sign of non-stationarity
#   Full-sample stationarity confirmed: 10/10 tenors pass ADF+KPSS dual gate
PCA_WINDOW     = 60           # rolling window for PCA fitting (trading days)
                              # 60d ≈ 2-3 mean reversion cycles — sufficient for covariance
                              # estimation. Keep responsive to regime changes.
                              # Revisit after level residual backtest.
N_COMPONENTS   = 2            # number of principal components to retain
                              # PC1 = parallel shift (~75-95% variance)
                              # PC2 = slope change   (~5-25% variance)
                              # Together explain 83-99% of yield curve variance
PCA_APPROACH   = 'level_volnorm'  # documentation flag — do not use in logic
                                  # 'level_volnorm' = PCA on vol-normalized yield levels
                                  # (replaces 'change' approach used prior to Sprint C-PROD-A)

# ── Z-Score ───────────────────────────────────────────────────────────────────
# Z-score is computed on level residuals (decimal yield units).
# Normalization window matches stationarity detection window (252d) so that
# the rolling mean and std used for Z-scoring are computed over the same
# horizon that confirms stationarity — consistent statistical assumptions.
ZSCORE_WINDOW       = 252     # rolling window for Z-score normalization
                              # Changed from 60d (change residuals) to 252d (level residuals)
                              # Level residuals require longer window — 15-25 day MR horizon
                              # needs 10+ cycles for reliable mean/std estimation
Z_ENTRY_THRESHOLD   = 2.0     # |Z| must exceed this to generate a signal
Z_EXIT_THRESHOLD    = 0.5     # mean reversion exit: LONG exits at Z <= +0.5, SHORT at Z >= -0.5
                              # Captures ~83% of mean reversion (Z=3.0 → Z=0.5)
                              # without waiting for full overshoot to opposite side
                              # Entry at Z=3.0, exit at Z=0.5 → 2.5σ of reversion captured
# Minimum cumulative explained variance by PC1+PC2.
# If PC1+PC2 cumulative variance drops below this threshold, signal is flagged.
# Default 0.80 — in normal regimes PC1+PC2 explains 83-99% of yield curve variance.
# Crisis periods can see this drop to 65-73% — below 0.80 PCA is not reliably
# capturing the curve and residuals may reflect noise rather than tradeable dislocations.
CUMULATIVE_VARIANCE_THRESHOLD = 0.80

# ── Stationarity ──────────────────────────────────────────────────────────────
# Level residuals require longer windows than change residuals because:
#   - Mean reversion horizon: 15-25 days (vs 1-5 days for change residuals)
#   - ADF needs 5-10 full MR cycles to have statistical power
#   - Minimum reliable window: 15d × 10 cycles = 150d → use 180d and 252d
#
# Window roles:
#   252d → entry gate (most reliable, 10/10 tenors TRADEABLE)
#   180d + 252d → exit vote (180d catches regime breaks 8-24 days earlier than 252d)
#   120d → display/escalation only (BORDERLINE for most tenors, not reliable for gates)
#   [10, 15, 20, 60] → retained for change residual reference, not used in production
ROLLING_ADF_WINDOWS      = [120, 180, 252]    # windows computed for level residuals
                                              # 120d: display/escalation only
                                              # 180d: exit vote
                                              # 252d: entry gate + exit vote
ADF_ENTRY_WINDOW         = 252                # entry gate window (changed from 60d)
                                              # 252d → 10/10 tenors TRADEABLE
                                              # 60d was too short (all AVOID for level residuals)
EXIT_ADF_WINDOWS         = [180, 252]         # exit vote windows (changed from [60])
                                              # Both must flag non-stationary to trigger exit
                                              # 180d catches breaks 8-24 days before 252d alone
                                              # vote_count == 0 means both [180d, 252d] flag
ADF_THRESHOLD            = 0.05               # ADF p-value threshold (unchanged)
KPSS_ENTRY_THRESHOLD     = 0.05               # KPSS p-value threshold (unchanged)
                                              # Entry requires: ADF p < 0.05 AND KPSS p > 0.05
                                              # Both tested on 252d rolling window
VOTE_EXIT_THRESHOLD      = 0                  # exit if vote_count == 0
                                              # For [180d, 252d]: both must flag non-stationary
# DEAD CODE — PC3 gate removed in Sprint D5. Kept for reference only. Do not use in logic.
PC3_ELEVATED_THRESHOLD   = 0.15               # PC3 variance threshold (unchanged)
                                              # Heuristic — refine after level residual backtest

# ── Risk & Execution ──────────────────────────────────────────────────────────
STOP_LOSS_BUFFER         = 2.0    # Z-score buffer above/below entry for stop loss
                                  # LONG stop: Z > entry_Z + 2.0 (e.g. Z > 5.0 at entry Z=3.0)
                                  # SHORT stop: Z < entry_Z - 2.0 (e.g. Z < -5.0 at entry Z=-3.0)
                                  # Widened from 1.5 for level residuals — larger magnitude
                                  # dislocations need more room before cutting loss
                                  # Revisit after backtest
REVERSION_ELIGIBLE_BUFFER = 0.5  # Z must move 0.5σ from entry before MR exit eligible
TRANSACTION_COST_BPS     = 0.25  # per leg (round-trip = 0.50 bps)
# Yield data from Federal Reserve is in decimal form (e.g. 0.0447 = 4.47%)
# Level residuals = actual_yield - PCA_reconstructed_yield (decimal units)
# Multiply by YIELD_TO_BPS_SCALAR to convert to basis points for P&L calculation.
#
# Level residual magnitudes at Z=3.0 (confirmed in PCA_Rollingtest.ipynb Cell 12):
#   Belly tenors:  9-13 bps per trade (18-26x round-trip cost coverage)
#   Short end:     21-27 bps per trade (42-54x cost coverage)
#   Long end:      21 bps per trade (42x cost coverage)
# This confirms the strategy is economically viable with 0.25 bps/leg costs.
YIELD_TO_BPS_SCALAR = 10_000
PAPER_TRADING            = True  # True = paper trading mode

# ── Position Sizing (duration-neutral, anchored to 10Yr = $1,000 notional) ───
NOTIONAL_MAP = {
    '1Mo':  100000,
    '3Mo':  35000,
    '6Mo':  17000,
    '1Yr':  9000,
    '2Yr':  4500,
    '3Yr':  3000,
    '5Yr':  2000,
    '7Yr':  1500,
    '10Yr': 1000,
    '30Yr': 500,
}

# ── DV01 Map ($ per bp per $1M notional) ─────────────────────────────────────
DV01_MAP = {
    '1Mo':  0.0083,
    '3Mo':  0.025,
    '6Mo':  0.049,
    '1Yr':  0.097,
    '2Yr':  0.190,
    '3Yr':  0.280,
    '5Yr':  0.450,
    '7Yr':  0.610,
    '10Yr': 0.850,
    '30Yr': 1.800,
}

# ── Tenor Ordering ────────────────────────────────────────────────────────────
TENORS = ['1Mo', '3Mo', '6Mo', '1Yr', '2Yr', '3Yr', '5Yr', '7Yr', '10Yr', '30Yr']

# ── Curve Segments ────────────────────────────────────────────────────────────
SEGMENT_MAP = {
    '1Mo': 'Short End', '3Mo': 'Short End', '6Mo': 'Short End', '1Yr': 'Short End',
    '2Yr': 'Belly Short', '3Yr': 'Belly Short', '5Yr': 'Belly Short',
    '7Yr': 'Belly Long', '10Yr': 'Belly Long',
    '30Yr': 'Long End',
}

SEGMENT_TENORS = {
    'Short End':   ['1Mo', '3Mo', '6Mo', '1Yr'],
    'Belly Short': ['2Yr', '3Yr', '5Yr'],
    'Belly Long':  ['7Yr', '10Yr'],
    'Long End':    ['30Yr'],
}

# ── Time Stop Buffers (trading days added to ACF horizon per segment) ─────────
# Updated for level residuals (Sprint C-PROD-A):
#   Change residuals: ACF horizon 1-5 days  → buffers of 3-12 days
#   Level residuals:  ACF horizon 15-25 days → buffers of 10-15 days
#   Total time stop = ACF horizon + buffer = 25-40 days for level residuals
#
# ACF horizons by segment (from level residual analysis):
#   Short End   (1Mo-1Yr):  ~15-21 days
#   Belly Short (2Yr-5Yr):  ~15-21 days
#   Belly Long  (7Yr-10Yr): ~16-18 days
#   Long End    (30Yr):     ~16 days
TIME_STOP_BUFFER_MAP = {
    'Short End':   10,    # was 3  — total stop ~25-31 days
    'Belly Short': 12,    # was 7  — total stop ~27-33 days
    'Belly Long':  12,    # was 10 — total stop ~28-30 days
    'Long End':    15,    # was 12 — total stop ~31 days
}

# ── Auction Suppression Windows (calendar days pre-auction) ───────────────────
AUCTION_SUPPRESS_DAYS = {
    'Bills':  2,
    'Notes':  5,
    'Bond':   7,
}

# ── Reporting ─────────────────────────────────────────────────────────────────
REPORT_OUTPUT_DIR = 'reports/output'


def config_validate():
    errors = []

    if not set(EXIT_ADF_WINDOWS).issubset(set(ROLLING_ADF_WINDOWS)):
        errors.append(
            f"EXIT_ADF_WINDOWS {EXIT_ADF_WINDOWS} must be a subset of "
            f"ROLLING_ADF_WINDOWS {ROLLING_ADF_WINDOWS}."
        )

    if Z_EXIT_THRESHOLD >= Z_ENTRY_THRESHOLD:
        errors.append(
            f"Z_EXIT_THRESHOLD ({Z_EXIT_THRESHOLD}) must be strictly less than "
            f"Z_ENTRY_THRESHOLD ({Z_ENTRY_THRESHOLD})."
        )

    if STOP_LOSS_BUFFER <= 0:
        errors.append(
            f"STOP_LOSS_BUFFER ({STOP_LOSS_BUFFER}) must be > 0."
        )

    if REVERSION_ELIGIBLE_BUFFER >= Z_ENTRY_THRESHOLD:
        errors.append(
            f"REVERSION_ELIGIBLE_BUFFER ({REVERSION_ELIGIBLE_BUFFER}) must be strictly less than "
            f"Z_ENTRY_THRESHOLD ({Z_ENTRY_THRESHOLD})."
        )

    if not (0.5 <= CUMULATIVE_VARIANCE_THRESHOLD <= 1.0):
        errors.append(
            f"CUMULATIVE_VARIANCE_THRESHOLD ({CUMULATIVE_VARIANCE_THRESHOLD}) must be between "
            f"0.5 and 1.0."
        )

    if ADF_ENTRY_WINDOW not in ROLLING_ADF_WINDOWS:
        errors.append(
            f"ADF_ENTRY_WINDOW ({ADF_ENTRY_WINDOW}) must be in "
            f"ROLLING_ADF_WINDOWS {ROLLING_ADF_WINDOWS}."
        )

    invalid_exit_windows = [w for w in EXIT_ADF_WINDOWS if w < 180]
    if invalid_exit_windows:
        errors.append(
            f"EXIT_ADF_WINDOWS contains values below 180: {invalid_exit_windows}. "
            f"Windows below 180 have insufficient statistical power for level residuals."
        )

    if errors:
        raise ValueError("config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    print("config validation passed")
