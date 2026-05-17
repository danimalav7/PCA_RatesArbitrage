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
PCA_WINDOW     = 60           # rolling window for PCA fitting (trading days)
N_COMPONENTS   = 2            # number of principal components to retain

# ── Z-Score ───────────────────────────────────────────────────────────────────
ZSCORE_WINDOW       = 60      # rolling window for Z-score normalization
Z_ENTRY_THRESHOLD   = 3.0     # |Z| must exceed this to generate a signal
Z_EXIT_THRESHOLD    = 1.0     # mean reversion exit: LONG exits at Z <= -1.0, SHORT at Z >= +1.0

# ── Stationarity ──────────────────────────────────────────────────────────────
ROLLING_ADF_WINDOWS      = [10, 15, 20, 60]   # windows for rolling ADF computation (display + escalation)
ADF_ENTRY_WINDOW         = 60                  # single window used for entry gate
EXIT_ADF_WINDOWS         = [60]               # windows used for exit vote — 60d only
                                              # Short windows (10d/15d) have insufficient statistical power
                                              # for exit decisions; kept in ROLLING_ADF_WINDOWS for
                                              # escalation display in signal card only
ADF_THRESHOLD            = 0.05   # ADF p-value threshold — reject unit root
KPSS_ENTRY_THRESHOLD     = 0.05   # KPSS p-value threshold — fail to reject stationarity
                                  # Entry requires: ADF p < 0.05 AND KPSS p > 0.05
VOTE_EXIT_THRESHOLD      = 0                  # exit if vote_count == 0 (all 4 windows flagging)
PC3_ELEVATED_THRESHOLD   = 0.15              # PC3 variance above this → elevated idiosyncratic risk
                                              # Rough 80th percentile heuristic — refine with full history

# ── Risk & Execution ──────────────────────────────────────────────────────────
STOP_LOSS_BUFFER         = 1.5    # Z-score buffer above/below entry for stop loss
REVERSION_ELIGIBLE_BUFFER = 0.5  # Z must move 0.5σ from entry before MR exit eligible
TRANSACTION_COST_BPS     = 0.25  # per leg (round-trip = 0.50 bps)
# Yield data from Federal Reserve is in decimal form (e.g. 0.0447 = 4.47%)
# PCA residuals are therefore in decimal yield units, not basis points.
# Multiply by YIELD_TO_BPS_SCALAR to convert residuals to basis points
# before applying DV01 × notional P&L formula.
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
TIME_STOP_BUFFER_MAP = {
    'Short End':   3,
    'Belly Short': 7,
    'Belly Long':  10,
    'Long End':    12,
}

# ── Auction Suppression Windows (calendar days pre-auction) ───────────────────
AUCTION_SUPPRESS_DAYS = {
    'Bills':  2,
    'Notes':  5,
    'Bond':   7,
}

# ── Reporting ─────────────────────────────────────────────────────────────────
REPORT_OUTPUT_DIR = 'reports/output'
