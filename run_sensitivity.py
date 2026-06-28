# ============================================================================
# run_sensitivity.py — Parameter sensitivity sweep
# ============================================================================
# Sweeps Z_ENTRY_THRESHOLD across a range of values and compares
# backtest performance metrics for each threshold.
#
# For each threshold value:
#   - Runs the full walk-forward backtest
#   - Records key performance metrics
#   - Produces terminal table, CSV, and HTML Plotly chart
#
# Usage:
#   python run_sensitivity.py
#
# Output:
#   reports/output/sensitivity_z_threshold.csv
#   reports/output/sensitivity_z_threshold.html
#
# Note: does not modify config.py — threshold is passed directly
# to run_backtest() by temporarily overriding config at runtime.
# config.py is restored to its original value after each run.
# ============================================================================

import copy
import datetime as dt
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
from utils.pipeline import setup_logging, build_pipeline_inputs
from backtest.engine import run_backtest
from data.auction import get_auction_suppression_flag

# ── Sweep parameters ──────────────────────────────────────────────────────────
# Z_ENTRY_THRESHOLD values to test. Current production value is 3.0.
# Lower values → more trades, lower conviction entries.
# Higher values → fewer trades, higher conviction entries.
Z_THRESHOLD_VALUES = [2.0, 2.25, 2.5, 2.75, 3.0]

# Metrics to collect per run
# All metrics computed from backtest_results dict
METRICS = [
    'z_threshold',
    'total_trades',
    'mr_exit_rate_pct',
    'sl_exit_rate_pct',
    'ns_exit_rate_pct',
    'ts_exit_rate_pct',
    'gross_pnl_bps',
    'net_pnl_bps',
    'avg_hold_days',
    'utilization_pct',
    'ann_excess_return_bps',
    'sharpe_10c',
]


def compute_metrics_from_results(
    z_threshold: float,
    backtest_results: dict,
) -> dict:
    """
    Extract key performance metrics from a backtest_results dict.

    Parameters
    ----------
    z_threshold : float
        The Z_ENTRY_THRESHOLD value used for this run.
    backtest_results : dict
        Output of run_backtest().

    Returns
    -------
    dict
        Flat dict of metrics matching METRICS list.
    """
    trade_log = pd.DataFrame(backtest_results['trade_log'])
    daily_pnl = backtest_results['daily_pnl']
    excess_returns_bps = backtest_results.get('excess_returns_bps')

    # Handle empty trade log
    if trade_log.empty:
        return {
            'z_threshold':          z_threshold,
            'total_trades':         0,
            'mr_exit_rate_pct':     0.0,
            'sl_exit_rate_pct':     0.0,
            'ns_exit_rate_pct':     0.0,
            'ts_exit_rate_pct':     0.0,
            'gross_pnl_bps':        0.0,
            'net_pnl_bps':          0.0,
            'avg_hold_days':        0.0,
            'utilization_pct':      0.0,
            'ann_excess_return_bps': 0.0,
            'sharpe_10c':           np.nan,
        }

    n = len(trade_log)

    # Exit reason rates
    exit_counts = trade_log['exit_reason'].value_counts()
    mr_rate  = exit_counts.get('MEAN-REVERSION', 0)  / n * 100
    sl_rate  = exit_counts.get('STOP-LOSS', 0)        / n * 100
    ns_rate  = exit_counts.get('NON-STATIONARY', 0)   / n * 100
    ts_rate  = exit_counts.get('TIME-STOP', 0)        / n * 100

    # P&L
    gross_pnl = (
        trade_log['pnl_bps'] + trade_log['transaction_cost_bps']
    ).sum()
    net_pnl = trade_log['pnl_bps'].sum()

    # Hold days
    avg_hold = trade_log['hold_days'].mean()

    # Utilization
    n_days = len(daily_pnl)
    active_days = (daily_pnl['TOTAL'] != 0).sum()
    utilization = active_days / n_days * 100

    # Section 10c overlay Sharpe
    if excess_returns_bps is not None and len(excess_returns_bps) > 0:
        n_years     = n_days / 252
        total_exc   = excess_returns_bps.sum()
        ann_excess  = total_exc / n_years
        mean_exc    = excess_returns_bps.mean()
        std_exc     = excess_returns_bps.std()
        sharpe_10c  = (
            (mean_exc / std_exc) * np.sqrt(252)
            if std_exc > 0 else np.nan
        )
    else:
        ann_excess = np.nan
        sharpe_10c = np.nan

    return {
        'z_threshold':           z_threshold,
        'total_trades':          n,
        'mr_exit_rate_pct':      round(mr_rate,  1),
        'sl_exit_rate_pct':      round(sl_rate,  1),
        'ns_exit_rate_pct':      round(ns_rate,  1),
        'ts_exit_rate_pct':      round(ts_rate,  1),
        'gross_pnl_bps':         round(gross_pnl, 2),
        'net_pnl_bps':           round(net_pnl,   2),
        'avg_hold_days':         round(avg_hold,  1),
        'utilization_pct':       round(utilization, 1),
        'ann_excess_return_bps': round(ann_excess, 2),
        'sharpe_10c':            round(sharpe_10c, 3),
    }


def print_comparison_table(results_list: list):
    """
    Print a formatted comparison table to terminal.

    Parameters
    ----------
    results_list : list of dict
        One dict per Z threshold value, output of
        compute_metrics_from_results().
    """
    SEP  = '=' * 100
    SEP2 = '-' * 100

    print(f"\n{SEP}")
    print(f"  SENSITIVITY SWEEP — Z_ENTRY_THRESHOLD")
    print(f"  Current production value: {config.Z_ENTRY_THRESHOLD}")
    print(f"  Values tested: {Z_THRESHOLD_VALUES}")
    print(f"{SEP}")

    # Header
    print(f"\n  {'Metric':<28}", end='')
    for r in results_list:
        print(f"  {'Z='+str(r['z_threshold']):>10}", end='')
    print()
    print(f"  {SEP2}")

    # Rows
    row_labels = {
        'total_trades':           'Total Trades',
        'mr_exit_rate_pct':       'MR Exit Rate (%)',
        'sl_exit_rate_pct':       'Stop-Loss Rate (%)',
        'ns_exit_rate_pct':       'NS Exit Rate (%)',
        'ts_exit_rate_pct':       'Time-Stop Rate (%)',
        'gross_pnl_bps':          'Gross P&L (bps)',
        'net_pnl_bps':            'Net P&L (bps)',
        'avg_hold_days':          'Avg Hold (days)',
        'utilization_pct':        'Utilization (%)',
        'ann_excess_return_bps':  'Ann. Excess Return (bps/yr)',
        'sharpe_10c':             'Sharpe Ratio (10c)',
    }

    for key, label in row_labels.items():
        print(f"  {label:<28}", end='')
        for r in results_list:
            val = r[key]
            if isinstance(val, float) and not np.isnan(val):
                print(f"  {val:>10.2f}", end='')
            elif isinstance(val, float) and np.isnan(val):
                print(f"  {'N/A':>10}", end='')
            else:
                print(f"  {val:>10}", end='')
        print()

    print(f"  {SEP2}")

    # Highlight best Sharpe
    sharpes = [r['sharpe_10c'] for r in results_list]
    valid   = [(s, r['z_threshold']) for s, r in
               zip(sharpes, results_list) if not np.isnan(s)]
    if valid:
        best_sharpe, best_z = max(valid, key=lambda x: x[0])
        print(f"\n  ★ Best Sharpe: {best_sharpe:.3f} at "
              f"Z_ENTRY_THRESHOLD = {best_z}")
        print(f"    Production value: Z = {config.Z_ENTRY_THRESHOLD}")
        if best_z != config.Z_ENTRY_THRESHOLD:
            print(f"    Consider updating config.py if "
                  f"Z={best_z} is consistently better across regimes.")
        else:
            print(f"    Current production value is optimal.")

    print(f"\n{SEP}\n")


def save_csv(results_list: list, output_dir: str) -> str:
    """
    Save sensitivity results to CSV.

    Parameters
    ----------
    results_list : list of dict
        One dict per Z threshold value.
    output_dir : str
        Directory to save CSV.

    Returns
    -------
    str
        Absolute path to saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results_list)
    path = os.path.join(output_dir, 'sensitivity_z_threshold.csv')
    df.to_csv(path, index=False)
    print(f"  CSV saved: {path}")
    return os.path.abspath(path)


def save_html_chart(results_list: list, output_dir: str) -> str:
    """
    Save a self-contained HTML Plotly comparison chart.

    Creates a 2x3 subplot grid showing:
      Row 1: Total Trades | Net P&L (bps) | Sharpe (10c)
      Row 2: MR Exit Rate | Stop-Loss Rate | Ann. Excess Return

    Parameters
    ----------
    results_list : list of dict
        One dict per Z threshold value.
    output_dir : str
        Directory to save HTML.

    Returns
    -------
    str
        Absolute path to saved HTML file.
    """
    df = pd.DataFrame(results_list)
    z_vals = df['z_threshold'].tolist()

    # Color production value bar differently
    prod_z = config.Z_ENTRY_THRESHOLD
    bar_colors = [
        '#2c7bb6' if z != prod_z else '#d7191c'
        for z in z_vals
    ]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Total Trades',
            'Net P&L (bps)',
            'Sharpe Ratio (10c)',
            'MR Exit Rate (%)',
            'Stop-Loss Rate (%)',
            'Ann. Excess Return (bps/yr)',
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    def bar_trace(y_col, row, col, color=None):
        colors = color or bar_colors
        fig.add_trace(
            go.Bar(
                x=[f'Z={z}' for z in z_vals],
                y=df[y_col].tolist(),
                marker_color=colors,
                showlegend=False,
                hovertemplate=(
                    f'Z=%{{x}}<br>{y_col}=%{{y:.2f}}<extra></extra>'
                ),
            ),
            row=row, col=col,
        )

    bar_trace('total_trades',          1, 1)
    bar_trace('net_pnl_bps',           1, 2)
    bar_trace('sharpe_10c',            1, 3)
    bar_trace('mr_exit_rate_pct',      2, 1)
    bar_trace('sl_exit_rate_pct',      2, 2)
    bar_trace('ann_excess_return_bps', 2, 3)

    # Add horizontal reference line on Sharpe chart
    # at the production Z value Sharpe
    prod_row = df[df['z_threshold'] == prod_z]
    if not prod_row.empty:
        prod_sharpe = float(prod_row['sharpe_10c'].iloc[0])
        fig.add_hline(
            y=prod_sharpe,
            row=1, col=3,
            line_dash='dash',
            line_color='#d7191c',
            line_width=1.5,
            annotation_text=f'Production (Z={prod_z})',
            annotation_position='top right',
            annotation_font=dict(color='#d7191c', size=10),
        )

    fig.update_layout(
        title=dict(
            text=(
                f'<b>Sensitivity Sweep — Z_ENTRY_THRESHOLD</b><br>'
                f'<sub>Red bar = current production value '
                f'(Z={prod_z}) | '
                f'Blue bars = alternative thresholds | '
                f'Generated: '
                f'{dt.datetime.now().strftime("%Y-%m-%d %H:%M")}'
                f'</sub>'
            ),
            x=0.5,
            font=dict(size=16),
        ),
        template='plotly_white',
        height=700,
        showlegend=False,
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'sensitivity_z_threshold.html')
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"  HTML chart saved: {path}")
    return os.path.abspath(path)


def main():
    log_path = setup_logging(log_dir='logs', log_file='sensitivity.log')

    print(f"\n{'='*60}")
    print(f"PCA RATES ARBITRAGE — SENSITIVITY SWEEP")
    print(f"Parameter: Z_ENTRY_THRESHOLD")
    print(f"Values:    {Z_THRESHOLD_VALUES}")
    print(f"Mode:      {config.MODE}")
    print(f"Started:   {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # Validate config before sweep
    config.config_validate()

    # ── Step 1: Build pipeline inputs once ────────────────────────────────
    # Pipeline inputs are the same for all threshold values —
    # only Z_ENTRY_THRESHOLD changes between runs.
    # Build once and reuse to avoid re-fetching data 5 times.
    print("Step 1: Building pipeline inputs (runs once for all thresholds)...")
    inputs = build_pipeline_inputs(
        start_date=config.START_DATE,
        end_date=None,
        mode=config.MODE,
    )
    print("  Pipeline inputs ready.\n")

    # Unpack all inputs needed by run_backtest()
    rates_data              = inputs['rates_data']
    residuals_df            = inputs['residuals_df']
    cumulative_variance_s   = inputs['cumulative_variance_s']
    rolling_adfs            = inputs['rolling_adfs']
    rolling_adf_252d        = inputs['rolling_adfs'][252]
    rolling_adf_180d        = inputs['rolling_adfs'][180]
    rolling_kpss_252d       = inputs['rolling_kpss_252d']
    rolling_kpss_180d       = inputs['rolling_kpss_180d']
    z_score_df              = inputs['z_score_df']
    acf_summary_df          = inputs['acf_summary_df']
    rolling_acf_horizons_df = inputs['rolling_acf_horizons_df']
    auction_calendar        = inputs['auction_calendar']

    # ── Step 2: Sweep Z_ENTRY_THRESHOLD ───────────────────────────────────
    print("Step 2: Running backtest for each threshold value...")
    print(f"  {'Threshold':<12} {'Trades':>8} {'NetBps':>10} "
          f"{'Sharpe':>8} {'Status'}")
    print(f"  {'-'*55}")

    results_list = []
    original_z   = config.Z_ENTRY_THRESHOLD

    for z_val in Z_THRESHOLD_VALUES:
        # Override Z_ENTRY_THRESHOLD at runtime
        # Does not modify config.py on disk
        config.Z_ENTRY_THRESHOLD = z_val

        try:
            backtest_results = run_backtest(
                residuals_df=residuals_df,
                z_score_df=z_score_df,
                rolling_adf_252d=rolling_adf_252d,
                rolling_kpss_252d=rolling_kpss_252d,
                rolling_adf_180d=rolling_adf_180d,
                rolling_kpss_180d=rolling_kpss_180d,
                rolling_adfs=rolling_adfs,
                cumulative_variance_s=cumulative_variance_s,
                rolling_acf_horizons_df=rolling_acf_horizons_df,
                rates_data=rates_data,
                acf_summary_df=acf_summary_df,
                auction_calendar=auction_calendar,
                get_auction_flag_fn=get_auction_suppression_flag,
                mode=config.MODE,
            )

            metrics = compute_metrics_from_results(z_val, backtest_results)
            results_list.append(metrics)

            print(f"  Z={z_val:<10}  "
                  f"{metrics['total_trades']:>8}  "
                  f"{metrics['net_pnl_bps']:>10.2f}  "
                  f"{metrics['sharpe_10c']:>8.3f}  ✓")

        except Exception as e:
            print(f"  Z={z_val:<10}  ERROR: {e}")
            results_list.append({
                'z_threshold': z_val,
                **{k: np.nan for k in METRICS if k != 'z_threshold'}
            })

        finally:
            # Always restore original value
            config.Z_ENTRY_THRESHOLD = original_z

    # Restore original value explicitly after loop
    config.Z_ENTRY_THRESHOLD = original_z
    print(f"\n  config.Z_ENTRY_THRESHOLD restored to {original_z}")

    # ── Step 3: Print comparison table ────────────────────────────────────
    print("\nStep 3: Printing comparison table...")
    print_comparison_table(results_list)

    # ── Step 4: Save CSV ───────────────────────────────────────────────────
    print("Step 4: Saving CSV...")
    csv_path = save_csv(results_list, config.REPORT_OUTPUT_DIR)

    # ── Step 5: Save HTML chart ────────────────────────────────────────────
    print("\nStep 5: Saving HTML chart...")
    html_path = save_html_chart(results_list, config.REPORT_OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✓ Sensitivity sweep complete")
    print(f"  Thresholds tested: {Z_THRESHOLD_VALUES}")
    print(f"  CSV:  {csv_path}")
    print(f"  HTML: {html_path}")
    print(f"  Log:  {log_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
