# ============================================================================
# utils/pipeline.py — Shared data pipeline for daily runner and backtest
# ============================================================================
# Provides build_pipeline_inputs() — shared Steps 1-5 called by both
# run_daily.py and run_backtest.py to avoid code duplication.
#
# Steps:
#   1. Fetch EOD rates from FRED
#   2. Compute rolling PCA residuals + cumulative variance Series
#   3. Compute rolling ADF stationarity windows (120/180/252d)
#   4. Compute rolling KPSS (252d and 180d)
#   5. Compute Z-scores
#   6. Compute ACF summary (full-sample, for signal cards)
#   7. Compute rolling 252d ACF horizons (regime-aware time stops)
#   8. Compute Hurst exponents (monitoring only)
#   9. Fetch auction calendar
# ============================================================================

import datetime as dt
import numpy as np
import pandas as pd
import logging
import config

from data.fetch_rates import FetchRates
from analytics.pca import compute_pca_residuals
from analytics.stationarity import (
    compute_rolling_adf,
    compute_rolling_kpss,
    compute_acf_summary,
    compute_rolling_acf_horizons,
    compute_hurst_exponent,
)
from signals.zscore import compute_zscore
from data.auction import fetch_auction_calendar

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = 'logs', log_file: str = 'pipeline.log'):
    """
    Configure structured logging to file and console.

    Creates log_dir if it does not exist.
    File handler: DEBUG level, appends to log_file.
    Console handler: INFO level.
    Format: timestamp | level | module | message

    Parameters
    ----------
    log_dir : str
        Directory for log files. Default: 'logs'.
    log_file : str
        Log filename. Default: 'pipeline.log'.
    """
    import os
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_file)

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler — DEBUG and above
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Avoid adding duplicate handlers if called multiple times
    if not root.handlers:
        root.addHandler(fh)
        root.addHandler(ch)

    logger.info("Logging initialised — file: %s", log_path)
    return log_path


def send_alert(subject: str, body: str, mode: str = config.MODE):
    """
    Send an email alert on pipeline failure or data anomaly.

    Reads SMTP configuration from environment variables:
        ALERT_EMAIL_FROM     : sender email address
        ALERT_EMAIL_TO       : recipient email address
        ALERT_EMAIL_PASSWORD : sender email password (app password)
        ALERT_SMTP_HOST      : SMTP host (default: smtp.gmail.com)
        ALERT_SMTP_PORT      : SMTP port (default: 587)

    If any environment variable is missing, logs a warning and
    skips sending — does not raise an exception.

    Parameters
    ----------
    subject : str
        Email subject line.
    body : str
        Email body text.
    mode : str
        Data mode label — included in subject for context.
    """
    import os
    import smtplib
    from email.mime.text import MIMEText

    required = [
        'ALERT_EMAIL_FROM',
        'ALERT_EMAIL_TO',
        'ALERT_EMAIL_PASSWORD',
    ]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        logger.warning(
            "Alert not sent — missing env vars: %s. "
            "Set these to enable email alerts.", missing
        )
        return

    from_addr = os.environ['ALERT_EMAIL_FROM']
    to_addr   = os.environ['ALERT_EMAIL_TO']
    password  = os.environ['ALERT_EMAIL_PASSWORD']
    smtp_host = os.environ.get('ALERT_SMTP_HOST', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('ALERT_SMTP_PORT', 587))

    full_subject = f"[PCA RatesArb | {mode}] {subject}"

    msg = MIMEText(body)
    msg['Subject'] = full_subject
    msg['From']    = from_addr
    msg['To']      = to_addr

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(from_addr, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        logger.info("Alert sent: %s", full_subject)
    except Exception as e:
        logger.error("Alert send failed: %s", str(e))


def build_pipeline_inputs(
    start_date: str = config.START_DATE,
    end_date: str = None,
    mode: str = config.MODE,
) -> dict:
    """
    Run the shared data pipeline and return all computed inputs.

    Called by both run_daily.py and run_backtest.py to avoid
    duplicating Steps 1-5 across both entry points.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
        Default: config.START_DATE.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. Default: today.
    mode : str
        Data mode. Default: config.MODE. Only 'EOD' implemented.

    Returns
    -------
    dict with keys:
        rates_data            : pd.DataFrame  — FetchRates() output
        residuals_df          : pd.DataFrame  — PCA level residuals
        cumulative_variance_s : pd.Series     — PC1+PC2 cumvar per date
        rolling_adfs          : dict          — {120:df, 180:df, 252:df}
        rolling_adf_252d      : pd.DataFrame  — rolling_adfs[252]
        rolling_adf_180d      : pd.DataFrame  — rolling_adfs[180]
        rolling_kpss_252d     : pd.DataFrame  — 252d KPSS p-values
        rolling_kpss_180d     : pd.DataFrame  — 180d KPSS p-values
        z_score_df            : pd.DataFrame  — rolling Z-scores
        acf_summary_df        : pd.DataFrame  — full-sample ACF summary
        rolling_acf_horizons_df : pd.DataFrame — rolling 252d ACF horizons
        hurst_df              : pd.DataFrame  — Hurst exponents
        auction_calendar      : pd.DataFrame  — TreasuryDirect calendar

    Raises
    ------
    RuntimeError
        If any pipeline step fails. Logs the error before raising.
    """

    logger.info("Pipeline starting — mode=%s start=%s", mode, start_date)

    try:
        # Step 1: Fetch rates
        logger.info("Step 1: Fetching EOD rates...")
        print("Step 1: Fetching EOD rates...")
        rates_data = FetchRates(
            start_date=start_date,
            end_date=end_date,
            mode=mode,
        )
        print(f"  Fetched {len(rates_data)} trading days "
              f"({rates_data['Date'].iloc[0]} → "
              f"{rates_data['Date'].iloc[-1]})")
        logger.info("Step 1 complete: %d trading days", len(rates_data))

        # Step 2: PCA residuals + cumulative variance
        logger.info("Step 2: Computing rolling PCA residuals...")
        print("\nStep 2: Computing rolling PCA residuals...")
        residuals_df, cumulative_variance_s = compute_pca_residuals(
            rates_data,
            window=config.PCA_WINDOW,
            n_components=config.N_COMPONENTS,
            mode=mode,
        )
        print(f"  Residuals computed: "
              f"{residuals_df.dropna().shape[0]} valid dates")
        logger.info("Step 2 complete: %d valid residual dates",
                    residuals_df.dropna().shape[0])

        # Step 3: Rolling ADF stationarity
        logger.info("Step 3: Computing rolling ADF...")
        print("\nStep 3: Computing rolling ADF stationarity windows...")
        rolling_adfs     = compute_rolling_adf(
            residuals_df,
            windows=config.ROLLING_ADF_WINDOWS,
            mode=mode,
        )
        rolling_adf_252d = rolling_adfs[252]
        rolling_adf_180d = rolling_adfs[180]
        logger.info("Step 3 complete")

        # Step 4: Rolling KPSS
        logger.info("Step 4: Computing rolling KPSS...")
        print("\nStep 4: Computing rolling KPSS...")
        rolling_kpss_252d = compute_rolling_kpss(
            residuals_df, window=252, mode=mode,
        )
        rolling_kpss_180d = compute_rolling_kpss(
            residuals_df, window=180, mode=mode,
        )
        logger.info("Step 4 complete")

        # Step 5: Z-scores
        logger.info("Step 5: Computing Z-scores...")
        print("\nStep 5: Computing Z-scores...")
        z_score_df = compute_zscore(
            residuals_df,
            window=config.ZSCORE_WINDOW,
            mode=mode,
        )
        logger.info("Step 5 complete")

        # Step 6: ACF summary (full-sample, for signal cards)
        logger.info("Step 6: Computing ACF summary and Hurst exponents...")
        print("\nStep 6: Computing ACF summary, rolling ACF horizons, "
              "Hurst exponents, and auction calendar...")
        acf_summary_df = compute_acf_summary(residuals_df)

        # Step 7: Rolling 252d ACF horizons (regime-aware time stops)
        rolling_acf_horizons_df = compute_rolling_acf_horizons(
            residuals_df, window=252, mode=mode,
        )

        # Step 8: Hurst exponents (monitoring only)
        hurst_df = compute_hurst_exponent(residuals_df)
        print(f"  Hurst exponents:")
        for tenor in config.TENORS:
            h = (hurst_df.loc[tenor, 'H']
                 if tenor in hurst_df.index else np.nan)
            print(f"    {tenor:<8} H={h:.3f}")
        logger.info("Step 6-8 complete")

        # Step 9: Auction calendar
        logger.info("Step 9: Fetching auction calendar...")
        auction_calendar = fetch_auction_calendar(mode=mode)
        logger.info("Step 9 complete: %d auction records",
                    len(auction_calendar))

        logger.info("Pipeline complete")

        return {
            'rates_data':              rates_data,
            'residuals_df':            residuals_df,
            'cumulative_variance_s':   cumulative_variance_s,
            'rolling_adfs':            rolling_adfs,
            'rolling_adf_252d':        rolling_adf_252d,
            'rolling_adf_180d':        rolling_adf_180d,
            'rolling_kpss_252d':       rolling_kpss_252d,
            'rolling_kpss_180d':       rolling_kpss_180d,
            'z_score_df':              z_score_df,
            'acf_summary_df':          acf_summary_df,
            'rolling_acf_horizons_df': rolling_acf_horizons_df,
            'hurst_df':                hurst_df,
            'auction_calendar':        auction_calendar,
        }

    except Exception as e:
        logger.error("Pipeline failed at step: %s", str(e), exc_info=True)
        raise RuntimeError(f"Pipeline failed: {e}") from e
