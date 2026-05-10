# ============================================================================
# reports/daily_report.py — Daily HTML report with Plotly charts
# ============================================================================
# Generates a self-contained HTML report for the daily signal scan.
# Output is written to config.REPORT_OUTPUT_DIR.
#
# Contents:
#   - Signal scan table (all 10 tenors, sorted by |Z|)
#   - Z-score time series chart (all tenors, last 252 days)
#   - Rolling 60d ADF p-value chart (all tenors)
#   - Stationarity escalation summary table
#   - Auction calendar suppression status
# ============================================================================

import os
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config


def generate_daily_report(
    signal_scan: pd.DataFrame,
    z_score_df: pd.DataFrame,
    rolling_adf_60d: pd.DataFrame,
    report_date: pd.Timestamp = None,
    output_dir: str = config.REPORT_OUTPUT_DIR,
    mode: str = config.MODE,
) -> str:
    """
    Generate a self-contained HTML daily report.

    Parameters
    ----------
    signal_scan : pd.DataFrame
        Output of scan_signals() for the report date.
    z_score_df : pd.DataFrame
        Full Z-score history. Used for time series chart.
    rolling_adf_60d : pd.DataFrame
        Full 60d ADF p-value history. Used for stationarity chart.
    report_date : pd.Timestamp, optional
        Date of the report. Defaults to today.
    output_dir : str
        Directory to write the HTML file.
        Default: config.REPORT_OUTPUT_DIR.
    mode : str
        Data mode label. Default: config.MODE.

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    if report_date is None:
        report_date = pd.Timestamp(dt.date.today())

    os.makedirs(output_dir, exist_ok=True)

    tenors = config.TENORS
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    color_map = dict(zip(tenors, colors))

    # ── Chart 1: Z-score time series (last 252 trading days) ─────────────────
    zscore_recent = z_score_df.tail(252)
    fig_zscore = go.Figure()
    for tenor in tenors:
        fig_zscore.add_trace(go.Scatter(
            x=zscore_recent.index,
            y=zscore_recent[tenor],
            mode='lines',
            name=tenor,
            line=dict(width=1.5, color=color_map[tenor]),
            hovertemplate=f'<b>{tenor}:</b> Z = %{{y:.2f}}<extra></extra>'
        ))
    fig_zscore.add_hline(
        y=config.Z_ENTRY_THRESHOLD,
        line_dash='dash', line_color='green', line_width=1.5,
        annotation_text=f'Z = +{config.Z_ENTRY_THRESHOLD} (LONG)',
        annotation_position='top right',
        annotation_font=dict(color='green', size=11)
    )
    fig_zscore.add_hline(
        y=-config.Z_ENTRY_THRESHOLD,
        line_dash='dash', line_color='red', line_width=1.5,
        annotation_text=f'Z = -{config.Z_ENTRY_THRESHOLD} (SHORT)',
        annotation_position='bottom right',
        annotation_font=dict(color='red', size=11)
    )
    fig_zscore.add_hline(
        y=0, line_dash='solid', line_color='gray',
        line_width=1, opacity=0.3
    )
    fig_zscore.update_layout(
        title=dict(
            text=(
                f'<b>Rolling {config.ZSCORE_WINDOW}-Day Z-Scores — '
                f'All Tenors</b><br>'
                f'Entry thresholds ±{config.Z_ENTRY_THRESHOLD} | '
                f'Last 252 trading days | Mode: {mode}'
            ),
            x=0.5, font=dict(size=16)
        ),
        xaxis_title='Date',
        yaxis_title='Z-Score (std devs)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h', yanchor='bottom',
            y=1.02, xanchor='center', x=0.5
        ),
        height=500,
    )

    # ── Chart 2: Rolling 60d ADF p-values ────────────────────────────────────
    adf_recent = rolling_adf_60d.tail(252)
    fig_adf = go.Figure()
    for tenor in tenors:
        fig_adf.add_trace(go.Scatter(
            x=adf_recent.index,
            y=adf_recent[tenor],
            mode='lines',
            name=tenor,
            line=dict(width=1.5, color=color_map[tenor]),
            hovertemplate=f'<b>{tenor}:</b> p = %{{y:.4f}}<extra></extra>'
        ))
    fig_adf.add_hline(
        y=config.ADF_THRESHOLD,
        line_dash='dash', line_color='red', line_width=1.5,
        annotation_text=f'p = {config.ADF_THRESHOLD}',
        annotation_position='top right',
        annotation_font=dict(color='red', size=11)
    )
    fig_adf.update_layout(
        title=dict(
            text=(
                f'<b>Rolling {config.ADF_ENTRY_WINDOW}-Day ADF p-Values — '
                f'All Tenors</b><br>'
                f'Below dashed line: stationary regime | '
                f'Last 252 trading days'
            ),
            x=0.5, font=dict(size=16)
        ),
        xaxis_title='Date',
        yaxis_title='ADF p-value',
        yaxis=dict(range=[0, 1.05], tickformat='.2f'),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h', yanchor='bottom',
            y=1.02, xanchor='center', x=0.5
        ),
        height=500,
    )

    # ── Signal scan table (HTML) ──────────────────────────────────────────────
    escalation_colors = {
        'CLEAR':         '#d4edda',
        'MONITOR':       '#fff3cd',
        'WARNING':       '#ffeeba',
        'DETERIORATING': '#f8d7da',
        'EXIT TRIGGER':  '#f5c6cb',
    }
    eligibility_colors = {
        'ELIGIBLE': '#d4edda',
        'WARNING':  '#fff3cd',
    }

    def _row_color(row):
        elig = str(row.get('trade_eligibility', ''))
        if elig == 'ELIGIBLE':
            return '#d4edda'
        if elig.startswith('WARNING'):
            return '#fff3cd'
        return '#f8d7da'

    table_rows = ''
    for _, row in signal_scan.iterrows():
        bg = _row_color(row)
        vote = (
            f"{int(row['stationarity_vote_count'])}/4"
            if not pd.isna(row.get('stationarity_vote_count')) else 'N/A'
        )
        pc3 = (
            f"{row['pc3_explained_variance']:.3f}"
            if not pd.isna(row.get('pc3_explained_variance')) else 'N/A'
        )
        auction = row.get('auction_flag', 'CLEAR')
        auction_str = (
            f"{auction} ({row.get('days_to_next_auction', '?')}d)"
            if auction != 'CLEAR' else 'CLEAR'
        )
        table_rows += f"""
        <tr style="background:{bg}">
            <td>{row['tenor']}</td>
            <td>{row['z_score']:.3f}</td>
            <td>{row.get('signal_direction', '')}</td>
            <td>{row.get('bond_status') or '—'}</td>
            <td>{row.get('rolling_adf_pvalue', np.nan):.4f}</td>
            <td>{vote}</td>
            <td>{row.get('stationarity_escalation', '')}</td>
            <td>{auction_str}</td>
            <td>{pc3}</td>
            <td style="font-size:0.85em">{row.get('trade_eligibility', '')}</td>
        </tr>"""

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    zscore_html = fig_zscore.to_html(
        full_html=False, include_plotlyjs='cdn'
    )
    adf_html = fig_adf.to_html(
        full_html=False, include_plotlyjs=False
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCA Rates Arbitrage — Daily Report {report_date.date()}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, sans-serif;
            margin: 0; padding: 20px;
            background: #f8f9fa; color: #212529;
        }}
        h1 {{ color: #343a40; border-bottom: 2px solid #dee2e6;
              padding-bottom: 10px; }}
        h2 {{ color: #495057; margin-top: 30px; }}
        .meta {{ color: #6c757d; font-size: 0.9em; margin-bottom: 20px; }}
        table {{
            border-collapse: collapse; width: 100%;
            background: white; border-radius: 8px;
            overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-size: 0.88em;
        }}
        th {{
            background: #343a40; color: white;
            padding: 10px 12px; text-align: left;
        }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #dee2e6; }}
        tr:last-child td {{ border-bottom: none; }}
        .chart-container {{ background: white; border-radius: 8px;
            padding: 10px; margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .legend {{
            display: flex; gap: 16px; flex-wrap: wrap;
            margin-bottom: 12px; font-size: 0.85em;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .swatch {{ width: 16px; height: 16px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>PCA Rates Arbitrage — Daily Report</h1>
    <div class="meta">
        <strong>Report date:</strong> {report_date.date()} &nbsp;|&nbsp;
        <strong>Mode:</strong> {mode} &nbsp;|&nbsp;
        <strong>Entry threshold:</strong> ±{config.Z_ENTRY_THRESHOLD} &nbsp;|&nbsp;
        <strong>Generated:</strong> {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>

    <h2>Signal Scan — All Tenors</h2>
    <div class="legend">
        <div class="legend-item">
            <div class="swatch" style="background:#d4edda"></div> Eligible
        </div>
        <div class="legend-item">
            <div class="swatch" style="background:#fff3cd"></div> Warning
        </div>
        <div class="legend-item">
            <div class="swatch" style="background:#f8d7da"></div> Blocked
        </div>
    </div>
    <table>
        <thead>
            <tr>
                <th>Tenor</th>
                <th>Z-Score</th>
                <th>Direction</th>
                <th>Status</th>
                <th>ADF p (60d)</th>
                <th>Vote</th>
                <th>Escalation</th>
                <th>Auction</th>
                <th>PC3</th>
                <th>Eligibility</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>

    <h2>Z-Score Time Series</h2>
    <div class="chart-container">{zscore_html}</div>

    <h2>Rolling ADF p-Values (60d)</h2>
    <div class="chart-container">{adf_html}</div>

</body>
</html>"""

    # ── Write file ────────────────────────────────────────────────────────────
    filename = (
        f"daily_report_{report_date.strftime('%Y%m%d')}.html"
    )
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Daily report written: {filepath}")
    return os.path.abspath(filepath)
