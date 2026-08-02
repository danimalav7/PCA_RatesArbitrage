# ============================================================================
# reports/daily_report.py — Unified daily HTML dashboard
# ============================================================================
# Generates a self-contained HTML dashboard updated every trading day.
#
# Section 1 — Live Signal Panel:
#   - Signal scan table (all 10 tenors, sorted by |Z|, color by eligibility)
#     Columns: Tenor, Z-Score, Direction, Status, ADF p (252d), Cumvar,
#              Vote, Escalation, Auction, Eligibility
#   - Z-score time series chart (last 252 trading days, Plotly)
#   - Rolling 252d ADF p-value chart (last 252 trading days, Plotly)
#
# Section 2 — Strategy Performance Panel:
#   - Only rendered if backtest_results pickle is loaded by run_daily.py
#   - KPI cards: Total Trades, Net P&L, Overall Sharpe (10b), MR Exit Rate
#   - P&L by tenor (Plotly horizontal bar, gross vs net)
#   - Sharpe table: Section 10c overlay IR — annualized excess return
#     and Sharpe ratio per tenor, all-days denominator, Fed Funds rf
#     on active days only (overlay on T-bill position)
#   - Exit reason breakdown (Plotly vertical bar)
#   - Year-by-year net P&L (Plotly vertical bar, green/red)
#   - Daily positions heatmap (last 252 trading days, Plotly)
#   - Top 5 trades table
#
# Output: self-contained HTML file at config.REPORT_OUTPUT_DIR/
#         daily_report_{YYYYMMDD}.html
# All Plotly charts use plotly_white template.
# ============================================================================

import math
import os
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import config


def _sharpe_bg(v):
    """Return background color for a Sharpe/IR value cell."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 'white'
    if np.isnan(f):
        return 'white'
    if f >= 2.0:
        return '#d4edda'
    if f >= 1.0:
        return '#fff3cd'
    if f >= 0:
        return '#f8d7da'
    return '#f5c6cb'


def generate_daily_report(
    signal_scan: pd.DataFrame,
    z_score_df: pd.DataFrame,
    rolling_adf_252d: pd.DataFrame,
    cumulative_variance_s: pd.Series,
    backtest_results: dict = None,
    report_date: pd.Timestamp = None,
    output_dir: str = config.REPORT_OUTPUT_DIR,
    mode: str = config.MODE,
) -> str:
    """
    Generate a self-contained HTML daily dashboard.

    Parameters
    ----------
    signal_scan : pd.DataFrame
        Output of scan_signals() for the report date.
    z_score_df : pd.DataFrame
        Full Z-score history. Used for time series chart.
        Last 252 trading days displayed.
    rolling_adf_252d : pd.DataFrame
        Full 252d rolling ADF p-value history.
        Last 252 trading days displayed.
    cumulative_variance_s : pd.Series
        PC1+PC2 cumulative explained variance per date.
        Output of compute_pca_residuals() second return value.
        Used in signal scan table replacing PC3 column.
    backtest_results : dict, optional
        Output of run_backtest(). If None, Section 2 is skipped
        and a placeholder message is shown instead.
        Keys used: trade_log, daily_pnl, daily_positions,
        validation_checks.
    report_date : pd.Timestamp, optional
        Date of the report. Defaults to today.
    output_dir : str
        Directory to write the HTML file.
    mode : str
        Data mode label.

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    if report_date is None:
        report_date = pd.Timestamp(dt.date.today())

    os.makedirs(output_dir, exist_ok=True)

    tenors = config.TENORS
    tenor_clr = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    color_map = dict(zip(tenors, tenor_clr))

    # ── CSS (regular string — curly braces are not f-string expressions) ──────
    css = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, sans-serif;
            margin: 0; padding: 20px;
            background: #f8f9fa; color: #212529;
        }
        h1 { color: #343a40; border-bottom: 2px solid #dee2e6;
              padding-bottom: 10px; }
        h2 { color: #495057; margin-top: 30px; }
        h3 { color: #495057; margin-top: 24px; }
        .meta { color: #6c757d; font-size: 0.9em; margin-bottom: 20px; }
        table {
            border-collapse: collapse; width: 100%;
            background: white; border-radius: 8px;
            overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-size: 0.88em;
        }
        th { background: #343a40; color: white;
             padding: 10px 12px; text-align: left; }
        td { padding: 8px 12px; border-bottom: 1px solid #dee2e6; }
        tr:last-child td { border-bottom: none; }
        .chart-container {
            background: white; border-radius: 8px;
            padding: 10px; margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .legend {
            display: flex; gap: 16px; flex-wrap: wrap;
            margin-bottom: 12px; font-size: 0.85em;
        }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .swatch { width: 16px; height: 16px; border-radius: 3px; }
        .section-divider {
            border-top: 3px solid #343a40; margin: 40px 0 20px 0;
        }
        .placeholder {
            background: #fff3cd; border: 1px solid #ffc107;
            border-radius: 8px; padding: 20px;
            margin: 20px 0; font-size: 1em;
        }
        .placeholder code {
            background: #f8f9fa; padding: 2px 6px;
            border-radius: 3px; font-family: monospace;
        }
        .sharpe-table {
            width: 100%; border-collapse: collapse; background: white;
            border-radius: 8px; overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-size: 0.88em; margin-bottom: 20px;
        }
        .sharpe-table th {
            background: #343a40; color: white;
            padding: 10px 12px; text-align: left;
        }
        .sharpe-table td { padding: 8px 12px; border-bottom: 1px solid #dee2e6; }
        .top-trades-table {
            width: 100%; border-collapse: collapse; background: white;
            border-radius: 8px; overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 0.88em;
        }
        .top-trades-table th {
            background: #343a40; color: white;
            padding: 10px 12px; text-align: left;
        }
        .top-trades-table td {
            padding: 8px 12px; border-bottom: 1px solid #dee2e6;
        }
        .metric-grid {
            display: grid; grid-template-columns: repeat(5, 1fr);
            gap: 16px; margin-bottom: 24px;
        }
        .metric-card {
            background: white; border-radius: 8px; padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;
        }
        .metric-card .value {
            font-size: 1.8em; font-weight: bold; color: #343a40;
        }
        .metric-card .label {
            font-size: 0.82em; color: #6c757d; margin-top: 4px;
        }
    """

    # ── Section 1, Chart 1: Z-score (include_plotlyjs='cdn') ─────────────────
    zscore_recent = z_score_df.tail(252)
    fig_z = go.Figure()
    for tenor in tenors:
        fig_z.add_trace(go.Scatter(
            x=zscore_recent.index, y=zscore_recent[tenor],
            mode='lines', name=tenor,
            line=dict(width=1.5, color=color_map[tenor]),
            hovertemplate=f'<b>{tenor}:</b> Z = %{{y:.2f}}<extra></extra>',
        ))
    fig_z.add_hline(
        y=config.Z_ENTRY_THRESHOLD, line_dash='dash',
        line_color='green', line_width=1.5,
        annotation_text=f'Z = +{config.Z_ENTRY_THRESHOLD} (LONG)',
        annotation_position='top right',
        annotation_font=dict(color='green', size=11),
    )
    fig_z.add_hline(
        y=-config.Z_ENTRY_THRESHOLD, line_dash='dash',
        line_color='red', line_width=1.5,
        annotation_text=f'Z = -{config.Z_ENTRY_THRESHOLD} (SHORT)',
        annotation_position='bottom right',
        annotation_font=dict(color='red', size=11),
    )
    fig_z.add_hline(
        y=0, line_dash='solid', line_color='gray', line_width=1, opacity=0.3,
    )
    fig_z.update_layout(
        title=dict(
            text=(
                f'<b>Rolling {config.ZSCORE_WINDOW}-Day Z-Scores — All Tenors</b>'
            ),
            x=0.5, font=dict(size=16),
        ),
        xaxis_title='Date', yaxis_title='Z-Score (std devs)',
        template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='center', x=0.5),
        height=500,
    )
    zscore_html = fig_z.to_html(full_html=False, include_plotlyjs='cdn')

    # ── Section 1, Chart 2: Rolling ADF p-values (include_plotlyjs=False) ────
    adf_recent = rolling_adf_252d.tail(252)
    fig_adf = go.Figure()
    for tenor in tenors:
        fig_adf.add_trace(go.Scatter(
            x=adf_recent.index, y=adf_recent[tenor],
            mode='lines', name=tenor,
            line=dict(width=1.5, color=color_map[tenor]),
            hovertemplate=f'<b>{tenor}:</b> p = %{{y:.4f}}<extra></extra>',
        ))
    fig_adf.add_hline(
        y=config.ADF_THRESHOLD, line_dash='dash',
        line_color='red', line_width=1.5,
        annotation_text='Stationary threshold (p=0.05)',
        annotation_position='top right',
        annotation_font=dict(color='red', size=11),
    )
    fig_adf.update_layout(
        title=dict(
            text=(
                f'<b>Rolling {config.ADF_ENTRY_WINDOW}-Day ADF p-Values — All Tenors</b>'
            ),
            x=0.5, font=dict(size=16),
        ),
        xaxis_title='Date', yaxis_title='ADF p-value',
        yaxis=dict(range=[0, 1.05], tickformat='.2f'),
        template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='center', x=0.5),
        height=500,
    )
    adf_html = fig_adf.to_html(full_html=False, include_plotlyjs=False)

    # ── Section 1: Signal scan table rows ─────────────────────────────────────
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
        vote_raw = row.get('stationarity_vote_count')
        vote = (
            f"{int(vote_raw)}/3"
            if vote_raw is not None and not pd.isna(vote_raw) else 'N/A'
        )
        cumvar_val = row.get('cumulative_variance', np.nan)
        try:
            cumvar_str = (
                f"{float(cumvar_val) * 100:.1f}%"
                if not pd.isna(cumvar_val) else 'N/A'
            )
        except (TypeError, ValueError):
            cumvar_str = 'N/A'
        auction = row.get('auction_flag', 'CLEAR')
        auction_str = (
            f"{auction} ({row.get('days_to_next_auction', '?')}d)"
            if auction != 'CLEAR' else 'CLEAR'
        )
        adf_p = row.get('rolling_adf_pvalue', np.nan)
        try:
            adf_str = f"{float(adf_p):.4f}" if not pd.isna(adf_p) else 'N/A'
        except (TypeError, ValueError):
            adf_str = 'N/A'
        z_thresh_val = row.get('z_threshold_used', config.Z_ENTRY_THRESHOLD)
        z_thresh_str = f"{z_thresh_val:.1f}"
        table_rows += f"""
            <tr style="background:{bg}">
                <td>{row['tenor']}</td>
                <td>{row['z_score']:.3f}</td>
                <td>{z_thresh_str}</td>
                <td>{row.get('signal_direction', '')}</td>
                <td>{row.get('bond_status') or '—'}</td>
                <td>{adf_str}</td>
                <td>{cumvar_str}</td>
                <td>{vote}</td>
                <td>{row.get('stationarity_escalation', '')}</td>
                <td>{auction_str}</td>
                <td style="font-size:0.85em">{row.get('trade_eligibility', '')}</td>
            </tr>"""

    # ── Section 2: conditional on backtest_results ────────────────────────────
    if backtest_results is None:
        section2_content = """
    <div class="placeholder">
        <p>&#9888; No backtest results found.</p>
        <p>Run <code>python run_backtest.py</code> to generate
           strategy performance metrics.</p>
    </div>"""
    else:
        tdf = pd.DataFrame(backtest_results.get('trade_log', []))
        daily_pnl      = backtest_results.get('daily_pnl', pd.DataFrame())
        daily_positions = backtest_results.get('daily_positions', pd.DataFrame())

        if len(tdf) == 0:
            section2_content = (
                '<p style="padding:20px;color:#666">'
                'No trades found in backtest results.</p>'
            )
        else:
            # Pre-compute fields
            if 'transaction_cost_bps' in tdf.columns:
                tdf['gross_pnl_bps'] = tdf['pnl_bps'] + tdf['transaction_cost_bps']
            else:
                tdf['gross_pnl_bps'] = tdf['pnl_bps']
            tdf['year'] = pd.to_datetime(tdf['entry_date']).dt.year

            n_trades = len(tdf)
            net_pnl  = round(float(tdf['pnl_bps'].sum()), 1)
            mr_rate  = f"{(tdf['exit_reason'] == 'MEAN-REVERSION').mean() * 100:.1f}%"

            # Extract Section 10c Sharpe for KPI cards
            sharpe_10c_kpi = 'N/A'
            ann_excess_kpi = 'N/A'
            exc_series = backtest_results.get('excess_returns_bps')
            if exc_series is not None and len(exc_series) > 0:
                n_days_bt  = len(exc_series)
                n_years_bt = n_days_bt / 252
                mean_e     = exc_series.mean()
                std_e      = exc_series.std()
                if std_e > 0:
                    sharpe_10c_kpi = f"{(mean_e / std_e) * np.sqrt(252):.3f}"
                total_e = exc_series.sum()
                if n_years_bt > 0:
                    ann_excess_kpi = f"{total_e / n_years_bt:.1f} bps/yr"

            # ── KPI metric cards (metric-grid) ────────────────────────────────
            kpi_html = f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="value">{n_trades}</div>
            <div class="label">Total Trades</div>
        </div>
        <div class="metric-card">
            <div class="value">{net_pnl}</div>
            <div class="label">Net P&amp;L (bps)</div>
        </div>
        <div class="metric-card">
            <div class="value">{ann_excess_kpi}</div>
            <div class="label">Ann. Excess Return</div>
        </div>
        <div class="metric-card">
            <div class="value">{sharpe_10c_kpi}</div>
            <div class="label">Sharpe (10c)</div>
        </div>
        <div class="metric-card">
            <div class="value">{mr_rate}</div>
            <div class="label">MR Exit Rate</div>
        </div>
    </div>"""

            # ── Chart A: P&L by tenor (horizontal bar) ────────────────────────
            tg = tdf.groupby('tenor').agg(
                gross=('gross_pnl_bps', 'sum'),
                net=('pnl_bps', 'sum'),
            )
            y_t      = [t for t in tenors if t in tg.index]
            gross_v  = [float(tg.loc[t, 'gross']) if t in tg.index else 0.0 for t in y_t]
            net_v    = [float(tg.loc[t, 'net'])   if t in tg.index else 0.0 for t in y_t]
            net_clrs = ['#2ca02c' if v > 0 else '#d62728' for v in net_v]

            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(
                x=gross_v, y=y_t, orientation='h', name='Gross P&L',
                marker_color='#4a90d9',
                hovertemplate='%{y}: %{x:.1f} bps<extra>Gross</extra>',
            ))
            fig_a.add_trace(go.Bar(
                x=net_v, y=y_t, orientation='h', name='Net P&L',
                marker_color=net_clrs,
                hovertemplate='%{y}: %{x:.1f} bps<extra>Net</extra>',
            ))
            fig_a.update_layout(
                title=dict(text='<b>P&L by Tenor — Gross vs Net (bps)</b>', x=0.5),
                xaxis_title='bps', barmode='group',
                height=450, template='plotly_white',
                legend=dict(orientation='h', y=1.02, x=0.5,
                            xanchor='center', yanchor='bottom'),
            )
            chart_a_html = fig_a.to_html(full_html=False, include_plotlyjs=False)

            # ── Sharpe table: Section 10c overlay IR ─────────────────────────
            def _fmt(v, dp=2):
                try:
                    f = float(v)
                    return 'N/A' if np.isnan(f) else f"{f:.{dp}f}"
                except (TypeError, ValueError):
                    return 'N/A'

            # Extract rates_data for FedFunds per-tenor lookup
            rates_ff_df = backtest_results.get('rates_data', None)
            if rates_ff_df is not None and 'FedFunds' in rates_ff_df.columns:
                rates_ff_idx = rates_ff_df.copy()
                rates_ff_idx['Date'] = pd.to_datetime(rates_ff_idx['Date'])
                rates_ff_idx = rates_ff_idx.set_index('Date')['FedFunds']
            else:
                rates_ff_idx = None

            n_days_total  = len(daily_pnl) if isinstance(daily_pnl, pd.DataFrame) else 0
            n_years_total = n_days_total / 252 if n_days_total > 0 else 1

            sharpe_rows = ''
            for t in tenors:
                sub = tdf[tdf['tenor'] == t]

                # Build per-tenor excess returns (Section 10c method)
                try:
                    if isinstance(daily_pnl, pd.DataFrame) and t in daily_pnl.columns:
                        tenor_pnl    = daily_pnl[t]
                        tenor_excess = pd.Series(0.0, index=daily_pnl.index)
                        active_mask  = (tenor_pnl != 0.0)
                        for date in daily_pnl.index:
                            if active_mask.loc[date]:
                                pnl = float(tenor_pnl.loc[date])
                                if rates_ff_idx is not None and date in rates_ff_idx.index:
                                    ff = float(rates_ff_idx.loc[date])
                                    ff = 0.0 if pd.isna(ff) else ff
                                else:
                                    ff = 0.0
                                rf_bps = (ff / 252) * config.YIELD_TO_BPS_SCALAR
                                tenor_excess.loc[date] = pnl - rf_bps
                        t_total   = tenor_excess.sum()
                        t_ann     = t_total / n_years_total
                        t_active  = int(active_mask.sum())
                        t_mean    = tenor_excess.mean()
                        t_std     = tenor_excess.std()
                        s10c      = (
                            (t_mean / t_std) * np.sqrt(252)
                            if t_std > 0 else np.nan
                        )
                    else:
                        t_total = t_ann = np.nan
                        t_active = 'N/A'
                        s10c = np.nan
                except Exception:
                    t_total = t_ann = np.nan
                    t_active = 'N/A'
                    s10c = np.nan

                sharpe_rows += f"""
            <tr>
                <td>{t}</td>
                <td>{t_active}</td>
                <td>{_fmt(t_ann, 1)}</td>
                <td>{_fmt(t_total, 1)}</td>
                <td style="background:{_sharpe_bg(s10c)}">{_fmt(s10c)}</td>
            </tr>"""

            # TOTAL row using portfolio-level excess_returns_bps
            if exc_series is not None and len(exc_series) > 0:
                tot_n_days  = len(exc_series)
                tot_n_years = tot_n_days / 252
                tot_total   = exc_series.sum()
                tot_ann     = tot_total / tot_n_years
                tot_active  = int((exc_series != 0).sum())
                tot_mean    = exc_series.mean()
                tot_std     = exc_series.std()
                tot_sharpe  = (
                    (tot_mean / tot_std) * np.sqrt(252)
                    if tot_std > 0 else np.nan
                )
            else:
                tot_ann = tot_total = np.nan
                tot_active = 'N/A'
                tot_sharpe = np.nan

            sharpe_rows += f"""
            <tr style="font-weight:bold;border-top:2px solid #343a40">
                <td>TOTAL</td>
                <td>{tot_active}</td>
                <td>{_fmt(tot_ann, 1)}</td>
                <td>{_fmt(tot_total, 1)}</td>
                <td style="background:{_sharpe_bg(tot_sharpe)}">{_fmt(tot_sharpe)}</td>
            </tr>"""

            sharpe_table_html = f"""
    <table class="sharpe-table">
        <thead>
            <tr>
                <th>Tenor</th><th>Active Days</th>
                <th>Ann. Excess (bps/yr)</th>
                <th>Total Excess (bps)</th>
                <th>Sharpe (10c)</th>
            </tr>
        </thead>
        <tbody>{sharpe_rows}</tbody>
    </table>"""

            # ── Chart B: Exit reason breakdown ────────────────────────────────
            reasons   = ['MEAN-REVERSION', 'NON-STATIONARY', 'TIME-STOP',
                         'STOP-LOSS', 'AUCTION-SUPPRESS']
            r_colors  = {
                'MEAN-REVERSION':   '#2ca02c',
                'NON-STATIONARY':   '#ff7f0e',
                'TIME-STOP':        '#9467bd',
                'STOP-LOSS':        '#d62728',
                'AUCTION-SUPPRESS': '#8c564b',
            }
            r_counts = [int((tdf['exit_reason'] == r).sum()) for r in reasons]
            total_r  = max(n_trades, 1)

            fig_b = go.Figure(go.Bar(
                x=reasons,
                y=r_counts,
                text=[f"{c} ({c / total_r * 100:.1f}%)" for c in r_counts],
                textposition='outside',
                marker_color=[r_colors[r] for r in reasons],
                hovertemplate='%{x}: %{y} trades<extra></extra>',
            ))
            fig_b.update_layout(
                title=dict(text='<b>Exit Reason Breakdown</b>', x=0.5),
                yaxis_title='Count',
                height=400, template='plotly_white',
            )
            chart_b_html = fig_b.to_html(full_html=False, include_plotlyjs=False)

            # ── Chart C: Year-by-year net P&L ─────────────────────────────────
            yr_pnl = tdf.groupby('year')['pnl_bps'].sum().sort_index()
            yr_clr = ['#2ca02c' if v > 0 else '#d62728' for v in yr_pnl.values]

            fig_c = go.Figure(go.Bar(
                x=[str(y) for y in yr_pnl.index],
                y=yr_pnl.values,
                text=[f"{v:.1f}" for v in yr_pnl.values],
                textposition='outside',
                marker_color=yr_clr,
                hovertemplate='%{x}: %{y:.1f} bps<extra></extra>',
            ))
            fig_c.update_layout(
                title=dict(text='<b>Year-by-Year Net P&L (bps)</b>', x=0.5),
                xaxis_title='Year', yaxis_title='Net bps',
                height=400, template='plotly_white',
            )
            chart_c_html = fig_c.to_html(full_html=False, include_plotlyjs=False)

            # ── Chart D: Daily Positions heatmap (last 252 trading days) ──────
            # Tenors: 1Mo at top, 30Yr at bottom.
            # Plotly heatmap y[0]=bottom → reverse tenor order so
            # 30Yr is y[0] (bottom) and 1Mo is y[-1] (top).
            if not daily_positions.empty:
                pos_rec  = daily_positions.tail(252)
                y_hm     = [t for t in reversed(tenors) if t in pos_rec.columns]
                x_dates  = [d.strftime('%Y-%m-%d') for d in pos_rec.index]
                pos_num  = pos_rec[y_hm].replace(
                    {'LONG': 1, 'SHORT': -1, 'No Position': 0}
                )
                z_vals   = pos_num.T.values.tolist()
                txt_vals = pos_rec[y_hm].T.values.tolist()
                n_dt     = len(x_dates)
                tickv    = [x_dates[i] for i in range(0, n_dt, 20)]

                fig_d = go.Figure(go.Heatmap(
                    x=x_dates, y=y_hm,
                    z=z_vals, text=txt_vals,
                    colorscale=[
                        [0,   '#d62728'],
                        [0.5, '#f8f9fa'],
                        [1,   '#2ca02c'],
                    ],
                    zmin=-1, zmax=1, showscale=False,
                    hovertemplate='%{y} | %{x} | %{text}<extra></extra>',
                ))
                fig_d.update_layout(
                    title=dict(
                        text='<b>Daily Positions — Last 252 Trading Days</b>',
                        x=0.5,
                    ),
                    height=350, template='plotly_white',
                    xaxis=dict(
                        tickangle=-45, tickmode='array',
                        tickvals=tickv, ticktext=tickv,
                    ),
                )
                chart_d_html = fig_d.to_html(full_html=False, include_plotlyjs=False)
            else:
                chart_d_html = (
                    '<p style="padding:20px;color:#666">'
                    'No position data available.</p>'
                )

            # ── Top 5 trades table ─────────────────────────────────────────────
            top5      = tdf.nlargest(5, 'pnl_bps')
            top5_rows = ''
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                dir_val = str(row.get('direction', ''))
                dir_bg  = '#d4edda' if dir_val == 'LONG' else '#ffeeba'
                ez_raw  = row.get('entry_zscore', np.nan)
                try:
                    ez_str = f"{float(ez_raw):.3f}" if not np.isnan(float(ez_raw)) else 'N/A'
                except (TypeError, ValueError):
                    ez_str = 'N/A'
                top5_rows += f"""
            <tr>
                <td>{i}</td>
                <td>{row.get('tenor', '')}</td>
                <td style="background:{dir_bg}">{dir_val}</td>
                <td>{str(row.get('entry_date', ''))[:10]}</td>
                <td>{str(row.get('exit_date', ''))[:10]}</td>
                <td>{row.get('exit_reason', '')}</td>
                <td>{ez_str}</td>
                <td>{float(row.get('pnl_bps', 0)):.1f}</td>
                <td>{int(row.get('hold_days', 0))}</td>
            </tr>"""

            top5_table_html = f"""
    <table class="top-trades-table">
        <thead>
            <tr>
                <th>#</th><th>Tenor</th><th>Direction</th>
                <th>Entry Date</th><th>Exit Date</th><th>Exit Reason</th>
                <th>Entry Z</th><th>Net bps</th><th>Hold Days</th>
            </tr>
        </thead>
        <tbody>{top5_rows}</tbody>
    </table>"""

            # Assemble Section 2 content
            section2_content = f"""
    {kpi_html}
    <h3>P&amp;L by Tenor</h3>
    <div class="chart-container">{chart_a_html}</div>
    <h3>Sharpe Ratios — Strategy Overlay IR (Section 10c)</h3>
    {sharpe_table_html}
    <h3>Exit Reason Breakdown</h3>
    <div class="chart-container">{chart_b_html}</div>
    <h3>Year-by-Year Net P&amp;L</h3>
    <div class="chart-container">{chart_c_html}</div>
    <h3>Daily Positions — Last 252 Trading Days</h3>
    <div class="chart-container">{chart_d_html}</div>
    <h3>Top 5 Trades</h3>
    {top5_table_html}"""

    # ── Assemble final HTML ───────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCA Rates Arbitrage — Daily Dashboard {report_date.date()}</title>
    <style>
{css}
    </style>
</head>
<body>
    <h1>PCA Rates Arbitrage — Daily Dashboard</h1>
    <div class="meta">
        <strong>Report date:</strong> {report_date.date()} &nbsp;|&nbsp;
        <strong>Mode:</strong> {mode} &nbsp;|&nbsp;
        <strong>Entry threshold:</strong> \xb1{config.Z_ENTRY_THRESHOLD} &nbsp;|&nbsp;
        <strong>Generated:</strong> {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>

    <!-- SECTION 1 -->
    <h2>Section 1 &mdash; Live Signal Panel</h2>

    <h3>Signal Scan &mdash; {report_date.date()}</h3>
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
                <th>Z Thresh</th>
                <th>Direction</th>
                <th>Status</th>
                <th>ADF p (252d)</th>
                <th>Cumvar</th>
                <th>Vote</th>
                <th>Escalation</th>
                <th>Auction</th>
                <th>Eligibility</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>

    <h3>Z-Score Time Series</h3>
    <div class="chart-container">{zscore_html}</div>

    <h3>Rolling ADF p-Values (252d)</h3>
    <div class="chart-container">{adf_html}</div>

    <!-- SECTION 2 -->
    <div class="section-divider"></div>
    <h2>Section 2 &mdash; Strategy Performance</h2>
    {section2_content}

</body>
</html>"""

    # ── Write file ────────────────────────────────────────────────────────────
    filename = f"daily_report_{report_date.strftime('%Y%m%d')}.html"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Daily report written: {filepath}")
    return os.path.abspath(filepath)
