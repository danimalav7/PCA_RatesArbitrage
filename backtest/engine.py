# ============================================================================
# backtest/engine.py — Walk-forward backtest engine and strategy diagnostics
# ============================================================================
# Provides:
#   run_backtest()             — full walk-forward P&L loop
#   run_strategy_diagnostics() — 10-section diagnostic report
#
# Design principles:
#   - All DataFrames passed explicitly — no global closures
#   - All parameters read from config — no hardcoding inside functions
#   - Look-ahead bias prevention: signal at T, execution at T+1
#   - Transaction costs deducted from daily_pnl on exit day
#
# BIAS PREVENTION RULES (NON-NEGOTIABLE):
# 1. Look-ahead bias: all signal computations on date T use data with index < T
# 2. Transaction timing: signals generated at close T, trades entered at open T+1
# 3. Z-score normalization: 60-day rolling mean/std excludes date T
# 4. Stale data: no signals if rates_data missing data for date T
# 5. No parameter optimization: all parameters fixed before backtest loop
#
# KNOWN ASSUMPTIONS & LIMITATIONS:
# A. EOD execution price: entry residual is recorded at T+1 EOD close, not T+1 open.
#    In reality, orders would execute at T+1 open. Using T+1 EOD is optimistic —
#    it gives the backtest the benefit of the full T+1 day's move before the
#    position is officially on. This assumption will be corrected when the IBKR
#    intraday feed is integrated (Sprint D), which will provide T+1 open prices.
# B. Yield data in decimal form: Federal Reserve yields are in decimal units
#    (e.g. 0.0447 = 4.47%). Residuals are therefore in decimal yield units.
#    YIELD_TO_BPS_SCALAR = 10,000 is applied when converting to basis points.
# C. Static DV01: DV01_MAP values are approximations at par. Actual DV01 drifts
#    as yields move. At current rate levels (4-5%), long-end DV01 may be
#    5-10% lower than the values in config.py.
# D. Level residual strategy: PCA on vol-normalized yield levels (Sprint C-PROD-A).
#    Residuals represent persistent yield mispricing (15-25 day MR horizon).
#    Z=3.0 corresponds to 9-27 bps of mispricing vs 0.50 bps round-trip cost.
#    Hurst H=0.80-0.86 — long memory, NOT non-stationarity.
#
# ENTRY CONDITIONS (all 4 must be true):
#   1. Signal threshold: |Z[T]| > config.Z_ENTRY_THRESHOLD (3.0) — signal at T
#   2. Dual stationarity gate at T (level residuals — Sprint C-PROD-A onwards):
#        rolling_adf_252d[T]  < config.ADF_THRESHOLD (0.05)  — rejects unit root
#        rolling_kpss_252d[T] > config.KPSS_ENTRY_THRESHOLD (0.05) — confirms stationarity
#        252d window required — level residuals need 10+ MR cycles for ADF power
#   3. No existing position for this tenor
#   4. Auction flag on T+1 must be CLEAR
#
# EXIT CHECKS (in priority order):
#   1. Stop loss: LONG exits if Z > entry_Z + 2.0; SHORT exits if Z < entry_Z - 2.0
#      Widened from 1.5 — level residuals have larger magnitude dislocations
#   2. Non-stationarity: get_stationarity_vote(T, windows=[180,252]) == 0
#      Both 180d AND 252d must flag non-stationary to trigger exit
#      180d catches regime breaks 8-24 days before 252d alone
#   3. Mean reversion: LONG exits if Z <= +0.5; SHORT exits if Z >= -0.5
#      Captures ~83% of mean reversion (Z=3.0 → Z=0.5)
#      Entry at Z=3.0, exit at Z=+0.5 → 2.5σ of reversion captured
#   4. Time stop: tenor-specific ACF horizon + segment buffer (25-40 days total)
#   5. Auction suppress (LONG and SHORT): auction_flag[T] == SUPPRESS
#
# P&L CONVENTION:
#   LONG  profits when residual DECREASES: pnl = -1 × residual_change
#   SHORT profits when residual INCREASES: pnl = +1 × residual_change
# ============================================================================

import numpy as np
import pandas as pd
import config
from analytics.stationarity import get_stationarity_vote


def run_backtest(
    residuals_df: pd.DataFrame,
    z_score_df: pd.DataFrame,
    rolling_adf_252d: pd.DataFrame,
    rolling_kpss_252d: pd.DataFrame,
    rolling_adf_180d: pd.DataFrame,
    rolling_kpss_180d: pd.DataFrame,
    rolling_adfs: dict,
    acf_summary_df: pd.DataFrame,
    auction_calendar: pd.DataFrame,
    get_auction_flag_fn: callable,
    mode: str = config.MODE,
) -> dict:
    """
    Run the walk-forward backtest for the PCA yield curve arbitrage strategy.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Daily PCA residuals. Output of compute_pca_residuals().
    z_score_df : pd.DataFrame
        Rolling Z-scores. Output of compute_zscore().
    rolling_adf_252d : pd.DataFrame
        252-day rolling ADF p-values. Used for entry gate only.
        Level residuals require 252d — 60d had insufficient statistical power (all AVOID).
    rolling_kpss_252d : pd.DataFrame
        252-day rolling KPSS p-values. Used alongside rolling_adf_252d for dual entry gate.
        Entry requires: ADF p < 0.05 AND KPSS p > 0.05 (both on 252d window).
    rolling_adf_180d : pd.DataFrame
        180-day rolling ADF p-values. Used for exit vote alongside 252d.
        180d catches regime breaks 8-24 days before 252d alone.
    rolling_kpss_180d : pd.DataFrame
        180-day rolling KPSS p-values. Used for exit vote alongside 252d.
        Exit fires when both 180d AND 252d flag non-stationary (vote_count == 0).
    rolling_adfs : dict
        All rolling ADF windows. Used for exit vote.
        Keys are window sizes (int).
    acf_summary_df : pd.DataFrame
        ACF/PACF summary. Output of compute_acf_summary().
    auction_calendar : pd.DataFrame
        Auction calendar DataFrame.
    get_auction_flag_fn : callable
        Signature: get_auction_flag_fn(date, tenor, auction_calendar, mode)
        -> (flag: str, info: dict)
    mode : str
        Data mode. Default: config.MODE.

    Returns
    -------
    dict with keys:
        'positions'         : dict of still-open positions at end of backtest
        'trade_log'         : list of completed trade dicts
        'daily_pnl'         : pd.DataFrame of daily P&L in bps per tenor
        'validation_checks' : dict of validation results
    """
    tenors         = config.TENORS
    segment_map    = config.SEGMENT_MAP
    segment_tenors = config.SEGMENT_TENORS
    dv01_map       = config.DV01_MAP
    notional_map   = config.NOTIONAL_MAP

    transaction_cost_bps      = config.TRANSACTION_COST_BPS
    z_entry_threshold         = config.Z_ENTRY_THRESHOLD        # 3.0
    z_exit_threshold          = config.Z_EXIT_THRESHOLD         # 0.5 — captures ~83% of reversion
    stop_loss_buffer          = config.STOP_LOSS_BUFFER         # 2.0 — widened for level residuals
    reversion_eligible_buffer = config.REVERSION_ELIGIBLE_BUFFER # 0.5
    adf_threshold             = config.ADF_THRESHOLD            # 0.05
    kpss_entry_threshold      = config.KPSS_ENTRY_THRESHOLD     # 0.05
    time_stop_buffer_map      = config.TIME_STOP_BUFFER_MAP     # 25-40 day total stops

    # ACF horizons per tenor
    acf_horizons_map = {}
    for tenor in tenors:
        if tenor in acf_summary_df.index:
            val = acf_summary_df.loc[tenor, 'First ACF Cross']
            acf_horizons_map[tenor] = int(val) if val != 'Never' else 20
        else:
            acf_horizons_map[tenor] = 20

    print(f"\n{'='*80}")
    print(f"BACKTEST INITIALIZATION — Mode: {mode}")
    print(f"{'='*80}")
    print(f"Entry threshold:    ±{z_entry_threshold}")
    print(f"Exit threshold:     ±{z_exit_threshold}")
    print(f"Stop loss buffer:   {stop_loss_buffer}σ from entry Z-score")
    print(f"Transaction costs:  {transaction_cost_bps} bps per leg "
          f"({transaction_cost_bps * 2} bps round-trip)")

    # ── Date alignment ────────────────────────────────────────────────────────
    # Date alignment uses 252d windows — first valid date is later than 60d approach
    # but ensures all entry/exit gate data is available on every backtest date
    available_dates = (
        set(rolling_adf_252d.index)
        & set(rolling_kpss_252d.index)
        & set(rolling_adf_180d.index)
        & set(rolling_kpss_180d.index)
        & set(z_score_df.index)
        & set(residuals_df.index)
    )
    backtest_dates = sorted(list(available_dates))

    if len(backtest_dates) == 0:
        raise ValueError("No overlapping dates across residuals, z_score, rolling_adf.")

    print(f"Backtest date range: {backtest_dates[0].date()} → "
          f"{backtest_dates[-1].date()}")
    print(f"Total trading days:  {len(backtest_dates)}")

    # ── State containers ──────────────────────────────────────────────────────
    positions = {}
    trade_log = []
    daily_pnl = pd.DataFrame(
        0.0, index=backtest_dates, columns=tenors + ['TOTAL']
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    for date_idx, current_date in enumerate(backtest_dates):
        execution_date_idx = date_idx + 1
        execution_date = (
            backtest_dates[execution_date_idx]
            if execution_date_idx < len(backtest_dates) else None
        )

        # ── SECTION A: EXIT CHECKS ────────────────────────────────────────────
        tenors_to_remove = []

        for tenor in tenors:
            if tenor not in positions:
                continue

            pos            = positions[tenor]
            direction      = pos['direction']
            entry_zscore   = pos['entry_zscore']
            entry_residual = pos['entry_residual_bps']
            time_stop_date = pos['time_stop_date']

            current_z = (
                float(z_score_df.loc[current_date, tenor])
                if current_date in z_score_df.index else np.nan
            )
            current_residual = (
                float(residuals_df.loc[current_date, tenor])
                if current_date in residuals_df.index else np.nan
            )

            is_entry_day = (current_date == pos['entry_date'])

            # Update reversion_eligible flag
            reversion_eligible = pos.get('reversion_eligible', False)
            if not reversion_eligible and not np.isnan(current_z):
                if (direction == 'LONG' and
                        current_z < (entry_zscore - reversion_eligible_buffer)):
                    reversion_eligible = True
                    pos['reversion_eligible'] = True
                elif (direction == 'SHORT' and
                        current_z > (entry_zscore + reversion_eligible_buffer)):
                    reversion_eligible = True
                    pos['reversion_eligible'] = True

            exit_triggered = False
            exit_reason    = None

            # Exit 1: Stop loss
            if not np.isnan(current_z):
                if (direction == 'LONG' and
                        current_z > (entry_zscore + stop_loss_buffer)):
                    exit_triggered = True
                    exit_reason    = 'STOP-LOSS'
                elif (direction == 'SHORT' and
                        current_z < (entry_zscore - stop_loss_buffer)):
                    exit_triggered = True
                    exit_reason    = 'STOP-LOSS'

            # Exit 2: Non-stationarity — [180d, 252d] windows both must flag
            # Both windows must agree (vote_count == 0 of 2) to trigger exit.
            # 180d catches regime breaks 8-24 days before 252d alone.
            # Using rolling_adfs which contains {120, 180, 252} — filter to EXIT_ADF_WINDOWS
            if not exit_triggered and not is_entry_day:
                vote_count = get_stationarity_vote(
                    current_date, tenor, rolling_adfs,
                    windows=config.EXIT_ADF_WINDOWS  # [180, 252]
                )
                if vote_count is not None and vote_count == 0:
                    exit_triggered = True
                    exit_reason    = 'NON-STATIONARY'

            # Exit 3: Mean reversion — captures ~83% of the signal
            # LONG entered at Z=+3.0 → exit when Z falls to +0.5 (2.5σ captured)
            # SHORT entered at Z=-3.0 → exit when Z rises to -0.5 (2.5σ captured)
            # z_exit_threshold = 0.5 (config.Z_EXIT_THRESHOLD)
            if (not exit_triggered and not is_entry_day
                    and reversion_eligible and not np.isnan(current_z)):
                if direction == 'LONG' and current_z <= z_exit_threshold:
                    exit_triggered = True
                    exit_reason    = 'MEAN-REVERSION'
                elif direction == 'SHORT' and current_z >= -z_exit_threshold:
                    exit_triggered = True
                    exit_reason    = 'MEAN-REVERSION'

            # Exit 4: Time stop
            if (not exit_triggered and not is_entry_day
                    and current_date >= time_stop_date):
                exit_triggered = True
                exit_reason    = 'TIME-STOP'

            # Exit 5: Auction suppress (LONG and SHORT)
            # Auction risk applies to both directions — an upcoming auction
            # disrupts yield levels regardless of position direction.
            if not exit_triggered and not is_entry_day:
                auction_flag, _ = get_auction_flag_fn(
                    current_date, tenor, auction_calendar, mode=mode
                )
                if auction_flag == 'SUPPRESS':
                    exit_triggered = True
                    exit_reason    = 'AUCTION-SUPPRESS'

            # ── Process exit ──────────────────────────────────────────────────
            if exit_triggered:
                # Accrue exit day P&L before closing position
                # NOTE: transaction costs are NOT deducted here — they are deducted
                # once only in pnl_bps below. Deducting here would double-count costs.
                if not np.isnan(current_residual):
                    exit_day_change = current_residual - pos['last_residual_bps']
                    exit_day_pnl_raw = exit_day_change * (
                        -1 if direction == 'LONG' else 1
                    )
                    exit_day_pnl_bps = exit_day_pnl_raw * config.YIELD_TO_BPS_SCALAR
                    pos['cumulative_pnl_raw'] += exit_day_pnl_raw
                    daily_pnl.loc[current_date, tenor] = exit_day_pnl_bps
                    pos['last_residual_bps'] = current_residual

                exit_z        = current_z if not np.isnan(current_z) else entry_zscore
                exit_residual = (
                    current_residual if not np.isnan(current_residual)
                    else entry_residual
                )
                hold_days   = (current_date - pos['entry_date']).days
                pnl_bps     = (pos['cumulative_pnl_raw'] * config.YIELD_TO_BPS_SCALAR) - transaction_cost_bps * 2
                pnl_dollars = (
                    pnl_bps
                    * dv01_map[tenor]
                    * notional_map[tenor] / 1e6
                    * 1000
                )

                trade_log.append({
                    'tenor':                  tenor,
                    'entry_date':             pos['entry_date'],
                    'entry_zscore':           entry_zscore,
                    'entry_residual_bps':     entry_residual,
                    'direction':              direction,
                    'stop_loss_zscore':       pos['stop_loss_zscore'],
                    'time_stop_date':         time_stop_date,
                    'auction_flag_at_entry':  pos.get('auction_flag_at_entry', 'UNKNOWN'),
                    'segment_at_entry':       pos.get('segment_at_entry', 'UNKNOWN'),
                    'exit_date':              current_date,
                    'exit_reason':            exit_reason,
                    'exit_zscore':            exit_z,
                    'exit_residual_bps':      exit_residual,
                    'hold_days':              hold_days,
                    'pnl_bps':                pnl_bps,
                    'pnl_dollars':            pnl_dollars,
                    'transaction_cost_bps':   transaction_cost_bps * 2,
                })
                tenors_to_remove.append(tenor)

        for tenor in tenors_to_remove:
            del positions[tenor]

        # ── SECTION B: ENTRY SIGNAL GENERATION ───────────────────────────────
        if execution_date is not None:
            for tenor in tenors:
                if tenor in positions:
                    continue

                # Signal check uses current_date (T) — no look-ahead bias
                # Execution residual uses execution_date (T+1) for entry price
                signal_z = (
                    float(z_score_df.loc[current_date, tenor])
                    if current_date in z_score_df.index else np.nan
                )
                # Entry gate: 252d ADF + KPSS dual confirmation
                # 252d required for level residuals — only window with reliable power
                signal_adf_p = (
                    float(rolling_adf_252d.loc[current_date, tenor])
                    if current_date in rolling_adf_252d.index else np.nan
                )
                signal_kpss_p = (
                    float(rolling_kpss_252d.loc[current_date, tenor])
                    if current_date in rolling_kpss_252d.index else np.nan
                )
                exec_residual = (
                    float(residuals_df.loc[execution_date, tenor])
                    if execution_date in residuals_df.index else np.nan
                )

                # Condition 1: Z threshold — checked at T (signal date), not T+1
                if np.isnan(signal_z) or abs(signal_z) <= z_entry_threshold:
                    continue

                direction = (
                    'LONG'  if signal_z >  z_entry_threshold else
                    'SHORT' if signal_z < -z_entry_threshold else None
                )
                if direction is None:
                    continue

                # Condition 2: Dual stationarity gate — both ADF and KPSS must confirm
                # ADF:  p < 0.05 → reject unit root → stationary evidence
                # KPSS: p > 0.05 → fail to reject stationarity → stationary evidence
                adf_pass  = not np.isnan(signal_adf_p)  and signal_adf_p  < adf_threshold
                kpss_pass = not np.isnan(signal_kpss_p) and signal_kpss_p > kpss_entry_threshold
                if not (adf_pass and kpss_pass):
                    continue

                # Condition 3: No existing position — checked by `if tenor in positions`

                # Condition 4: Auction flag
                seg_name = segment_map[tenor]
                auction_flag, _ = get_auction_flag_fn(
                    execution_date, tenor, auction_calendar, mode=mode
                )
                if auction_flag != 'CLEAR':
                    continue

                # Compute stop loss and time stop
                stop_loss_z = (
                    signal_z + stop_loss_buffer if direction == 'LONG'
                    else signal_z - stop_loss_buffer
                )
                time_stop = execution_date + pd.Timedelta(
                    days=acf_horizons_map[tenor]
                    + time_stop_buffer_map[seg_name]
                )

                positions[tenor] = {
                    'entry_date':           execution_date,
                    'entry_zscore':         signal_z,
                    'entry_residual_bps':   (
                        exec_residual if not np.isnan(exec_residual) else 0.0
                    ),
                    'direction':            direction,
                    'stop_loss_zscore':     stop_loss_z,
                    'time_stop_date':       time_stop,
                    'auction_flag_at_entry': auction_flag,
                    'segment_at_entry':     seg_name,
                    'reversion_eligible':   False,
                    'last_residual_bps':    (
                        exec_residual if not np.isnan(exec_residual) else 0.0
                    ),
                    'cumulative_pnl_raw':   0.0,
                }

        # ── SECTION C: DAILY P&L ACCUMULATION ────────────────────────────────
        for tenor in positions:
            pos       = positions[tenor]
            direction = pos['direction']
            current_residual = (
                float(residuals_df.loc[current_date, tenor])
                if current_date in residuals_df.index else 0.0
            )

            residual_change   = current_residual - pos['last_residual_bps']
            daily_pnl_raw     = residual_change * (-1 if direction == 'LONG' else 1)
            daily_pnl_bps     = daily_pnl_raw * config.YIELD_TO_BPS_SCALAR
            pos['last_residual_bps']    = current_residual
            pos['cumulative_pnl_raw']  += daily_pnl_raw
            daily_pnl.loc[current_date, tenor] = daily_pnl_bps

        daily_pnl.loc[current_date, 'TOTAL'] = (
            daily_pnl.loc[current_date, tenors].sum()
        )

    # ── VALIDATION CHECKS ─────────────────────────────────────────────────────
    validation_checks = {}
    n_trades_opened = len(trade_log)
    n_trades_open   = len(positions)
    validation_checks['trades_opened'] = n_trades_opened
    validation_checks['trades_closed'] = n_trades_opened - n_trades_open
    validation_checks['trades_open']   = n_trades_open
    validation_checks['check_1_warning'] = n_trades_open > 10

    days_with_position = (daily_pnl[tenors].abs() > 0).any(axis=1).sum()
    utilization_pct    = days_with_position / len(backtest_dates) * 100
    validation_checks['utilization_pct']  = utilization_pct
    validation_checks['check_2_warning']  = utilization_pct > 80

    if trade_log:
        min_entry_z = min(abs(t['entry_zscore']) for t in trade_log)
        validation_checks['min_entry_zscore'] = min_entry_z
        validation_checks['check_3_warning']  = min_entry_z < (
            z_entry_threshold - 0.1
        )
    else:
        validation_checks['min_entry_zscore'] = np.nan
        validation_checks['check_3_warning']  = False

    exit_reasons = {}
    for trade in trade_log:
        r = trade['exit_reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1
    validation_checks['exit_reasons'] = exit_reasons

    if 'STOP-LOSS' in exit_reasons and trade_log:
        sl_pct = exit_reasons['STOP-LOSS'] / len(trade_log) * 100
        validation_checks['stop_loss_pct']   = sl_pct
        validation_checks['check_4_warning'] = sl_pct > 50
    else:
        validation_checks['stop_loss_pct']   = 0.0
        validation_checks['check_4_warning'] = False

    # First 3 trades walkthrough
    walkthroughs = []
    for trade_idx, trade in enumerate(trade_log[:3]):
        tenor      = trade['tenor']
        entry_date = trade['entry_date']
        exit_date  = trade['exit_date']
        mask = (
            (residuals_df.index >= entry_date) &
            (residuals_df.index <= exit_date)
        )
        residuals_in_trade = residuals_df.loc[mask, tenor]

        wt = {
            'trade_num':      trade_idx + 1,
            'tenor':          tenor,
            'direction':      trade['direction'],
            'entry_date':     entry_date.strftime('%Y-%m-%d'),
            'entry_zscore':   round(trade['entry_zscore'], 3),
            'entry_residual': round(trade['entry_residual_bps'], 4),
            'exit_date':      exit_date.strftime('%Y-%m-%d'),
            'exit_reason':    trade['exit_reason'],
            'exit_residual':  round(trade['exit_residual_bps'], 4),
            'hold_days':      trade['hold_days'],
            'pnl_bps':        round(trade['pnl_bps'], 4),
            'pnl_dollars':    round(trade['pnl_dollars'], 2),
            'daily_steps':    [],
        }
        for i, res_date in enumerate(residuals_in_trade.index):
            prev_res = (
                trade['entry_residual_bps'] if i == 0
                else float(residuals_in_trade.iloc[i - 1])
            )
            curr_res = float(residuals_in_trade.iloc[i])
            change   = curr_res - prev_res
            daily_pnl_contribution = change * (
                -1 if trade['direction'] == 'LONG' else 1
            ) * config.YIELD_TO_BPS_SCALAR
            wt['daily_steps'].append({
                'date':      res_date.strftime('%Y-%m-%d'),
                'residual':  round(curr_res, 4),
                'change':    round(change,   4),
                'daily_pnl': round(daily_pnl_contribution, 4),
            })
        walkthroughs.append(wt)
    validation_checks['first_trades_walkthrough'] = walkthroughs

    # ── PRINT VALIDATION RESULTS ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"VALIDATION CHECKS")
    print(f"{'='*80}\n")

    print(f"Check 1: Position open/close balance")
    print(f"  Trades opened:    {validation_checks['trades_opened']}")
    print(f"  Trades closed:    {validation_checks['trades_closed']}")
    print(f"  Trades still open:{validation_checks['trades_open']}")
    print(f"  {'✓ PASS' if not validation_checks['check_1_warning'] else '⚠ WARNING: >10 unclosed'}")

    print(f"\nCheck 2: Strategy utilization")
    print(f"  Days with ≥1 open position: {days_with_position} / "
          f"{len(backtest_dates)} ({utilization_pct:.1f}%)")
    print(f"  {'✓ PASS' if not validation_checks['check_2_warning'] else '⚠ WARNING: >80%'}")

    print(f"\nCheck 3: Minimum entry Z-score magnitude")
    if not np.isnan(validation_checks['min_entry_zscore']):
        print(f"  Minimum |Z| at entry: "
              f"{validation_checks['min_entry_zscore']:.3f}")
        print(f"  {'✓ PASS' if not validation_checks['check_3_warning'] else '⚠ WARNING: below threshold'}")
    else:
        print(f"  No trades to check")

    print(f"\nCheck 4: Exit reason distribution")
    if trade_log:
        for reason in sorted(exit_reasons):
            count = exit_reasons[reason]
            pct   = count / len(trade_log) * 100
            print(f"  {reason:<20}: {count:>4} ({pct:>5.1f}%)")
        print(f"  {'✓ PASS' if not validation_checks['check_4_warning'] else '⚠ WARNING: STOP-LOSS >50%'}")

    print(f"\nCheck 5: First 3 closed trades P&L walkthrough")
    for wt in walkthroughs:
        print(f"\n  Trade {wt['trade_num']}: {wt['tenor']} {wt['direction']}")
        print(f"    Entry: {wt['entry_date']} @ Z={wt['entry_zscore']:>7.3f}, "
              f"Residual={wt['entry_residual']:>8.4f} bps")
        print(f"    Exit:  {wt['exit_date']} ({wt['exit_reason']}) @ "
              f"Residual={wt['exit_residual']:>8.4f} bps")
        print(f"    Hold:  {wt['hold_days']} days | "
              f"P&L: {wt['pnl_bps']:>8.4f} bps | "
              f"${wt['pnl_dollars']:>10.2f}")
        for step in wt['daily_steps']:
            print(f"      {step['date']}: Residual={step['residual']:>8.4f}, "
                  f"Change={step['change']:>8.4f}, "
                  f"DailyP&L={step['daily_pnl']:>8.4f}")
    if walkthroughs:
        print(f"\n  ✓ First 3 trades manual verification complete")

    print(f"\n{'='*80}")
    print(f"⏸  VALIDATION COMPLETE — review before proceeding to metrics")
    print(f"{'='*80}\n")

    return {
        'positions':         positions,
        'trade_log':         trade_log,
        'daily_pnl':         daily_pnl,
        'validation_checks': validation_checks,
    }


def run_strategy_diagnostics(
    backtest_results: dict,
    dv01_map: dict     = config.DV01_MAP,
    notional_map: dict = config.NOTIONAL_MAP,
    z_entry_threshold: float = config.Z_ENTRY_THRESHOLD,
) -> None:
    """
    Print 10-section strategy diagnostic report.

    Designed to be re-run after any parameter change without re-running
    the backtest. All inputs come from backtest_results dict.

    Parameters
    ----------
    backtest_results : dict
        Output of run_backtest().
    dv01_map : dict
        DV01 per tenor. Default: config.DV01_MAP.
    notional_map : dict
        Notional per tenor. Default: config.NOTIONAL_MAP.
    z_entry_threshold : float
        Entry Z-score threshold used in the backtest. Default: config.Z_ENTRY_THRESHOLD.
    """
    trade_log_df = pd.DataFrame(backtest_results['trade_log'])
    daily_pnl    = backtest_results['daily_pnl']

    if trade_log_df.empty:
        print("No trades to analyze.")
        return

    tenors_order = config.TENORS
    exit_reasons = [
        'MEAN-REVERSION', 'NON-STATIONARY', 'TIME-STOP',
        'STOP-LOSS', 'AUCTION-SUPPRESS'
    ]

    # Pre-compute P&L fields
    trade_log_df['gross_pnl_bps'] = (
        trade_log_df['pnl_bps'] + trade_log_df['transaction_cost_bps']
    )
    trade_log_df['dv01']     = trade_log_df['tenor'].map(dv01_map)
    trade_log_df['notional'] = trade_log_df['tenor'].map(notional_map)
    trade_log_df['gross_pnl_usd'] = (
        trade_log_df['gross_pnl_bps']
        * trade_log_df['dv01']
        * (trade_log_df['notional'] / 1e6) * 1000
    )
    trade_log_df['net_pnl_usd'] = (
        trade_log_df['pnl_bps']
        * trade_log_df['dv01']
        * (trade_log_df['notional'] / 1e6) * 1000
    )
    trade_log_df['year'] = pd.to_datetime(
        trade_log_df['entry_date']
    ).dt.year

    # Daily dollar P&L
    daily_pnl_dollars = pd.Series(0.0, index=daily_pnl.index)
    for tenor in tenors_order:
        if tenor in daily_pnl.columns:
            daily_pnl_dollars += (
                daily_pnl[tenor]
                * dv01_map[tenor]
                * (notional_map[tenor] / 1e6) * 1000
            )

    SEP  = '=' * 80
    SEP2 = '-' * 80

    # ── Section 1: Total Trades ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SECTION 1 — TOTAL TRADES SUMMARY")
    print(f"{SEP}")
    print(f"  {'Total trades':<40} {len(trade_log_df):>10}")
    print(f"  {'Z-score threshold':<40} ±{z_entry_threshold:>9.1f}")
    print(f"\n  {'Tenor':<8}  {'Total':>6}  {'LONG':>6}  {'SHORT':>6}  "
          f"{'% of Total':>10}")
    print(f"  {SEP2}")
    for tenor in tenors_order:
        sub   = trade_log_df[trade_log_df['tenor'] == tenor]
        longs = (sub['direction'] == 'LONG').sum()
        shrts = (sub['direction'] == 'SHORT').sum()
        pct   = len(sub) / len(trade_log_df) * 100 if len(trade_log_df) > 0 else 0
        print(f"  {tenor:<8}  {len(sub):>6}  {longs:>6}  {shrts:>6}  "
              f"{pct:>9.1f}%")
    total_l = (trade_log_df['direction'] == 'LONG').sum()
    total_s = (trade_log_df['direction'] == 'SHORT').sum()
    print(f"  {SEP2}")
    print(f"  {'TOTAL':<8}  {len(trade_log_df):>6}  "
          f"{total_l:>6}  {total_s:>6}")

    # ── Section 2 & 3: Gross and Net P&L by Tenor ────────────────────────────
    print(f"\n{SEP}")
    print(f"  SECTION 2 & 3 — GROSS AND NET P&L BY TENOR")
    print(f"{SEP}")
    print(f"  {'Tenor':<8}  {'GrossBps':>9}  {'GrossUSD':>10}  "
          f"{'NetBps':>9}  {'NetUSD':>10}")
    print(f"  {SEP2}")
    for tenor in tenors_order:
        sub = trade_log_df[trade_log_df['tenor'] == tenor]
        if sub.empty:
            continue
        print(f"  {tenor:<8}  "
              f"{sub['gross_pnl_bps'].sum():>9.2f}  "
              f"${sub['gross_pnl_usd'].sum():>9,.0f}  "
              f"{sub['pnl_bps'].sum():>9.2f}  "
              f"${sub['net_pnl_usd'].sum():>9,.0f}")
    print(f"  {SEP2}")
    print(f"  {'TOTAL':<8}  "
          f"{trade_log_df['gross_pnl_bps'].sum():>9.2f}  "
          f"${trade_log_df['gross_pnl_usd'].sum():>9,.0f}  "
          f"{trade_log_df['pnl_bps'].sum():>9.2f}  "
          f"${trade_log_df['net_pnl_usd'].sum():>9,.0f}")

    # ── Section 4: Exit Reasons by Tenor ─────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SECTION 4 — EXIT REASONS BY TENOR (%)")
    print(f"{SEP}")
    reasons_short = ['MR%', 'NS%', 'TS%', 'SL%', 'AUC%']
    header = f"  {'Tenor':<8}" + "".join(
        f"  {r:>6}" for r in reasons_short
    ) + f"  {'N':>5}"
    print(header)
    print(f"  {SEP2}")
    for tenor in tenors_order:
        sub = trade_log_df[trade_log_df['tenor'] == tenor]
        if sub.empty:
            continue
        row = f"  {tenor:<8}"
        for reason in exit_reasons:
            pct = (sub['exit_reason'] == reason).mean() * 100
            row += f"  {pct:>5.1f}%"
        row += f"  {len(sub):>5}"
        print(row)

    # ── Section 5: LONG vs SHORT by Tenor ────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SECTION 5 — LONG vs SHORT TRADES BY TENOR")
    print(f"{SEP}")
    print(f"  {'Tenor':<8}  {'LONG':>6}  {'SHORT':>6}  {'L%':>6}  {'S%':>6}")
    print(f"  {SEP2}")
    for tenor in tenors_order:
        sub   = trade_log_df[trade_log_df['tenor'] == tenor]
        if sub.empty:
            continue
        longs = (sub['direction'] == 'LONG').sum()
        shrts = (sub['direction'] == 'SHORT').sum()
        print(f"  {tenor:<8}  {longs:>6}  {shrts:>6}  "
              f"{longs/len(sub)*100:>5.1f}%  "
              f"{shrts/len(sub)*100:>5.1f}%")
    print(f"  {SEP2}")
    print(f"  {'TOTAL':<8}  {total_l:>6}  {total_s:>6}  "
          f"{total_l/len(trade_log_df)*100:>5.1f}%  "
          f"{total_s/len(trade_log_df)*100:>5.1f}%")

    # ── Section 6: Profitable Trades by Direction ─────────────────────────────
    print(f"\n{SEP}")
    print(f"  SECTION 6 — PROFITABLE TRADES BY DIRECTION AND TENOR")
    print(f"{SEP}")
    print(f"  {'Tenor':<8}  {'L+ve':>6}  {'LTot':>6}  {'L%':>7}  "
          f"{'S+ve':>6}  {'STot':>6}  {'S%':>7}")
    print(f"  {SEP2}")
    for tenor in tenors_order:
        sub   = trade_log_df[trade_log_df['tenor'] == tenor]
        if sub.empty:
            continue
        l_sub = sub[sub['direction'] == 'LONG']
        s_sub = sub[sub['direction'] == 'SHORT']
        l_pos = (l_sub['pnl_bps'] > 0).sum()
        s_pos = (s_sub['pnl_bps'] > 0).sum()
        l_pct = l_pos / len(l_sub) * 100 if len(l_sub) > 0 else 0
        s_pct = s_pos / len(s_sub) * 100 if len(s_sub) > 0 else 0
        print(f"  {tenor:<8}  {l_pos:>6}  {len(l_sub):>6}  {l_pct:>6.1f}%  "
              f"{s_pos:>6}  {len(s_sub):>6}  {s_pct:>6.1f}%")

    # ── Section 7: Hold Days by Tenor ─────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SECTION 7 — HOLD DAYS BY TENOR")
    print(f"{SEP}")
    print(f"  {'Tenor':<8}  {'Avg':>7}  {'Median':>7}  {'Min':>5}  "
          f"{'Max':>5}  {'MR Avg':>8}  {'TS Avg':>8}")
    print(f"  {SEP2}")
    for tenor in tenors_order:
        sub    = trade_log_df[trade_log_df['tenor'] == tenor]
        if sub.empty:
            continue
        mr_sub = sub[sub['exit_reason'] == 'MEAN-REVERSION']
        ts_sub = sub[sub['exit_reason'] == 'TIME-STOP']
        mr_avg = mr_sub['hold_days'].mean() if not mr_sub.empty else np.nan
        ts_avg = ts_sub['hold_days'].mean() if not ts_sub.empty else np.nan
        print(f"  {tenor:<8}  "
              f"{sub['hold_days'].mean():>7.1f}  "
              f"{sub['hold_days'].median():>7.1f}  "
              f"{sub['hold_days'].min():>5}  "
              f"{sub['hold_days'].max():>5}  "
              f"{mr_avg:>8.1f}  "
              f"{ts_avg:>8.1f}")
    print(f"  {SEP2}")
    print(f"  {'TOTAL':<8}  "
          f"{trade_log_df['hold_days'].mean():>7.1f}  "
          f"{trade_log_df['hold_days'].median():>7.1f}  "
          f"{trade_log_df['hold_days'].min():>5}  "
          f"{trade_log_df['hold_days'].max():>5}")

    # ── Section 8: LONG vs SHORT by Year ─────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SECTION 8 — LONG vs SHORT TRADES BY YEAR")
    print(f"{SEP}")
    print(f"  {'Year':<6}  {'LONG':>6}  {'SHORT':>6}  {'Total':>6}  "
          f"{'NetBps':>9}  {'NetUSD':>10}")
    print(f"  {SEP2}")
    for year in sorted(trade_log_df['year'].unique()):
        sub   = trade_log_df[trade_log_df['year'] == year]
        longs = (sub['direction'] == 'LONG').sum()
        shrts = (sub['direction'] == 'SHORT').sum()
        print(f"  {year:<6}  {longs:>6}  {shrts:>6}  {len(sub):>6}  "
              f"{sub['pnl_bps'].sum():>9.2f}  "
              f"${sub['net_pnl_usd'].sum():>9,.0f}")
    print(f"  {SEP2}")
    print(f"  {'TOTAL':<6}  {total_l:>6}  {total_s:>6}  "
          f"{len(trade_log_df):>6}  "
          f"{trade_log_df['pnl_bps'].sum():>9.2f}  "
          f"${trade_log_df['net_pnl_usd'].sum():>9,.0f}")

    # ── Section 9: Top 5 Trades ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SECTION 9 — TOP 5 TRADES BY NET P&L")
    print(f"{SEP}")
    top5 = trade_log_df.nlargest(5, 'pnl_bps')
    print(f"  {'#':<3}  {'Tenor':<6}  {'Dir':<6}  {'Entry':<12}  "
          f"{'Exit':<12}  {'Reason':<16}  {'Z':>7}  "
          f"{'NetBps':>8}  {'NetUSD':>10}  {'Hold':>5}")
    print(f"  {SEP2}")
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"  {rank:<3}  {row['tenor']:<6}  {row['direction']:<6}  "
              f"{str(row['entry_date'])[:10]:<12}  "
              f"{str(row['exit_date'])[:10]:<12}  "
              f"{row['exit_reason']:<16}  "
              f"{row['entry_zscore']:>7.2f}  "
              f"{row['pnl_bps']:>8.4f}  "
              f"${row['net_pnl_usd']:>9,.0f}  "
              f"{row['hold_days']:>5}")

    # ── Section 10: Sharpe by Tenor ───────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SECTION 10 — SHARPE RATIO BY TENOR (annualized, rf=0)")
    print(f"{SEP}")
    print(f"  {'Tenor':<8}  {'Sharpe':>8}  {'AvgDailyUSD':>13}  "
          f"{'StdDailyUSD':>13}")
    print(f"  {SEP2}")
    for tenor in tenors_order:
        if tenor not in daily_pnl.columns:
            continue
        tenor_daily_usd = (
            daily_pnl[tenor]
            * dv01_map[tenor]
            * (notional_map[tenor] / 1e6) * 1000
        )
        # Deduct costs on exit days
        tenor_trades = trade_log_df[trade_log_df['tenor'] == tenor]
        for _, tr in tenor_trades.iterrows():
            exit_d = tr['exit_date']
            if exit_d in tenor_daily_usd.index:
                cost_usd = (
                    tr['transaction_cost_bps']
                    * dv01_map[tenor]
                    * (notional_map[tenor] / 1e6) * 1000
                )
                tenor_daily_usd.loc[exit_d] -= cost_usd
        active = tenor_daily_usd[tenor_daily_usd != 0]
        if len(active) < 2 or active.std() == 0:
            sharpe = np.nan
        else:
            sharpe = (active.mean() / active.std()) * np.sqrt(252)
        print(f"  {tenor:<8}  {sharpe:>8.3f}  "
              f"${active.mean():>12,.4f}  "
              f"${active.std():>12,.4f}")
    print(f"  {SEP2}")
    overall_sharpe = (
        (daily_pnl_dollars.mean() / daily_pnl_dollars.std()) * np.sqrt(252)
        if daily_pnl_dollars.std() > 0 else np.nan
    )
    print(f"  {'OVERALL':<8}  {overall_sharpe:>8.3f}")
    print(f"\n{SEP}\n")
