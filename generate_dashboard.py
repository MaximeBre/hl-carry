"""
generate_dashboard.py – HTML dashboard for HL Carry System

Generates outputs/dashboard.html with Chart.js visualisations.
Reads from:
  - outputs/performance_log.csv  (hourly PnL)
  - state/state.json             (current positions)
  - state/trades.json            (trade log)
  - outputs/backtest_*.json      (per-asset backtest results)
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR   = Path("outputs")
STATE_PATH   = Path("state/state.json")
TRADES_PATH  = Path("state/trades.json")
PERF_LOG     = OUTPUT_DIR / "performance_log.csv"
DASHBOARD    = OUTPUT_DIR / "dashboard.html"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path, default=None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def _load_perf_log() -> pd.DataFrame:
    if PERF_LOG.exists():
        try:
            return pd.read_csv(PERF_LOG, parse_dates=["timestamp"])
        except Exception:
            pass
    return pd.DataFrame()


def _load_backtest_results() -> dict:
    """Load per-asset backtest JSON files from outputs/."""
    results = {}
    for p in OUTPUT_DIR.glob("backtest_*.json"):
        symbol = p.stem.replace("backtest_", "")
        try:
            with open(p) as f:
                results[symbol] = json.load(f)
        except Exception:
            pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Chart.js dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def _series_to_chartjs(index, values, label: str, color: str) -> dict:
    return {
        "label": label,
        "data": [
            {"x": str(t)[:19], "y": round(float(v), 6) if v == v else None}
            for t, v in zip(index, values)
        ],
        "borderColor": color,
        "backgroundColor": color + "22",
        "borderWidth": 1.5,
        "pointRadius": 0,
        "tension": 0.1,
        "fill": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_dashboard() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state       = _load_json(STATE_PATH, {})
    trades      = _load_json(TRADES_PATH, [])
    perf_df     = _load_perf_log()
    bt_results  = _load_backtest_results()

    # ── Prepare PnL chart data ────────────────────────────────────────────
    pnl_datasets = []
    if not perf_df.empty and "portfolio_value" in perf_df.columns:
        pnl_datasets.append(_series_to_chartjs(
            perf_df["timestamp"], perf_df["portfolio_value"],
            "Portfolio Value (USDT)", "#58a6ff",
        ))

    # Per-asset predicted rates
    pred_datasets = []
    symbols = ["BTC", "ETH", "SOL", "AVAX", "XRP"]
    colors  = ["#f0883e", "#3fb950", "#a371f7", "#ffa657", "#79c0ff"]
    for sym, col in zip(symbols, colors):
        col_name = f"pred_{sym}"
        if not perf_df.empty and col_name in perf_df.columns:
            pred_datasets.append(_series_to_chartjs(
                perf_df["timestamp"], perf_df[col_name] * 100,
                f"{sym} Pred Rate (%)", col,
            ))

    # ── Current positions table ───────────────────────────────────────────
    positions = state.get("positions", {})
    pos_rows = ""
    for sym in symbols:
        p = positions.get(sym, {})
        s = p.get("state", "–")
        sz = p.get("position_size", 0.0)
        entry = (p.get("entry_time") or "–")[:19]
        pred_e = p.get("entry_prediction", 0.0)
        held = p.get("hours_held", 0)
        state_color = {
            "HOLDING":  "#3fb950",
            "ENTERING": "#ffa657",
            "EXITING":  "#f85149",
            "FLAT":     "#8b949e",
        }.get(s, "#8b949e")
        pos_rows += f"""
        <tr>
          <td>{sym}</td>
          <td><span style="color:{state_color};font-weight:bold">{s}</span></td>
          <td>{sz:.2f}</td>
          <td>{entry}</td>
          <td>{pred_e*100:.4f}%</td>
          <td>{held}h</td>
        </tr>"""

    # ── Recent trades table ───────────────────────────────────────────────
    recent_trades = trades[-20:][::-1]  # latest 20, newest first
    trade_rows = ""
    for t in recent_trades:
        pnl = t.get("price", 0)  # re-used field for paper fills
        trade_rows += f"""
        <tr>
          <td>{t.get('ts','')[:19]}</td>
          <td>{t.get('symbol','')}</td>
          <td>{t.get('exchange','')}</td>
          <td>{t.get('side','')}</td>
          <td>{t.get('qty',0):.4f}</td>
          <td>{t.get('price',0):.2f}</td>
          <td>{'PAPER' if t.get('paper') else 'LIVE'}</td>
        </tr>"""

    # ── Backtest summary table ────────────────────────────────────────────
    bt_rows = ""
    for sym, bt in bt_results.items():
        m = bt.get("metrics", {})
        valid = "✓" if m.get("statistically_valid") else "✗"
        bt_rows += f"""
        <tr>
          <td>{sym}</td>
          <td>{m.get('sharpe_ratio',0):.3f}</td>
          <td>{m.get('total_return',0)*100:.2f}%</td>
          <td>{m.get('max_drawdown',0)*100:.2f}%</td>
          <td>{m.get('trade_count',0)}</td>
          <td>{m.get('win_rate',0)*100:.1f}%</td>
          <td>{m.get('time_in_market_pct',0)*100:.1f}%</td>
          <td>{valid}</td>
        </tr>"""

    # ── Portfolio summary ─────────────────────────────────────────────────
    portfolio_val = state.get("portfolio_value", 1000.0)
    run_count     = state.get("run_count", 0)
    last_update   = (state.get("last_update") or "–")[:19]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>HL Carry – Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0d1117;color:#e6edf3;font-family:system-ui,-apple-system,sans-serif;padding:24px}}
    h1{{color:#58a6ff;font-size:1.6rem;margin-bottom:4px}}
    .subtitle{{color:#8b949e;font-size:.85rem;margin-bottom:24px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:28px}}
    .card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
    .card .label{{color:#8b949e;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}}
    .card .value{{color:#e6edf3;font-size:1.6rem;font-weight:700;margin-top:4px}}
    .card .value.positive{{color:#3fb950}}
    .card .value.negative{{color:#f85149}}
    .chart-wrap{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin-bottom:24px}}
    .chart-wrap h2{{color:#8b949e;font-size:.9rem;margin-bottom:12px;text-transform:uppercase;letter-spacing:.06em}}
    canvas{{max-height:300px}}
    table{{width:100%;border-collapse:collapse;font-size:.82rem}}
    th{{color:#8b949e;text-align:left;padding:8px 12px;border-bottom:1px solid #30363d;font-weight:500}}
    td{{padding:7px 12px;border-bottom:1px solid #21262d;color:#e6edf3}}
    tr:hover td{{background:#161b22}}
    .section-title{{color:#58a6ff;font-size:1rem;margin:28px 0 12px;font-weight:600}}
    .badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75rem;font-weight:600}}
    .badge.paper{{background:#1f2a1f;color:#3fb950;border:1px solid #3fb950}}
  </style>
</head>
<body>
  <h1>HL Carry System</h1>
  <div class="subtitle">
    Last update: {last_update} UTC &nbsp;|&nbsp;
    Run #{run_count} &nbsp;|&nbsp;
    <span class="badge paper">PAPER TRADING</span>
  </div>

  <!-- Summary cards -->
  <div class="grid">
    <div class="card">
      <div class="label">Portfolio Value</div>
      <div class="value">${portfolio_val:.2f}</div>
    </div>
    <div class="card">
      <div class="label">P&L</div>
      <div class="value {'positive' if portfolio_val >= 1000 else 'negative'}">
        ${portfolio_val - 1000:.2f}
      </div>
    </div>
    <div class="card">
      <div class="label">Return</div>
      <div class="value {'positive' if portfolio_val >= 1000 else 'negative'}">
        {(portfolio_val / 1000 - 1) * 100:.2f}%
      </div>
    </div>
    <div class="card">
      <div class="label">Open Positions</div>
      <div class="value">
        {sum(1 for p in positions.values() if p.get('state') == 'HOLDING')}
      </div>
    </div>
    <div class="card">
      <div class="label">Total Fills</div>
      <div class="value">{len(trades)}</div>
    </div>
  </div>

  <!-- PnL chart -->
  <div class="chart-wrap">
    <h2>Portfolio Value Over Time</h2>
    <canvas id="pnlChart"></canvas>
  </div>

  <!-- Predicted rates chart -->
  <div class="chart-wrap">
    <h2>Predicted 24h Cumulative Rate (%) per Asset</h2>
    <canvas id="predChart"></canvas>
  </div>

  <!-- Current positions -->
  <p class="section-title">Current Positions</p>
  <div class="chart-wrap">
    <table>
      <thead>
        <tr>
          <th>Symbol</th><th>State</th><th>Size (USDT)</th>
          <th>Entry Time</th><th>Entry Pred.</th><th>Held</th>
        </tr>
      </thead>
      <tbody>{pos_rows}</tbody>
    </table>
  </div>

  <!-- Backtest summary -->
  <p class="section-title">Backtest Results (Walk-Forward)</p>
  <div class="chart-wrap">
    <table>
      <thead>
        <tr>
          <th>Symbol</th><th>Sharpe</th><th>Total Return</th>
          <th>Max DD</th><th>Trades</th><th>Win Rate</th>
          <th>Time in Market</th><th>Valid ≥30</th>
        </tr>
      </thead>
      <tbody>{bt_rows if bt_rows else '<tr><td colspan="8" style="color:#8b949e;text-align:center">No backtest results yet – run backtest first</td></tr>'}</tbody>
    </table>
  </div>

  <!-- Recent fills -->
  <p class="section-title">Recent Paper Fills (last 20)</p>
  <div class="chart-wrap">
    <table>
      <thead>
        <tr>
          <th>Timestamp</th><th>Symbol</th><th>Exchange</th>
          <th>Side</th><th>Qty</th><th>Price</th><th>Mode</th>
        </tr>
      </thead>
      <tbody>{trade_rows if trade_rows else '<tr><td colspan="7" style="color:#8b949e;text-align:center">No fills yet</td></tr>'}</tbody>
    </table>
  </div>

  <script>
  const COLORS = ["#58a6ff","#f0883e","#3fb950","#a371f7","#ffa657","#79c0ff"];

  function makeChart(id, datasets, yLabel, xType='timeseries') {{
    const ctx = document.getElementById(id);
    if (!ctx) return;
    new Chart(ctx, {{
      type: 'line',
      data: {{ datasets }},
      options: {{
        responsive: true,
        interaction: {{ mode: 'index', intersect: false }},
        plugins: {{
          legend: {{ labels: {{ color: '#8b949e', boxWidth: 12 }} }},
          tooltip: {{ backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1,
                      titleColor: '#e6edf3', bodyColor: '#8b949e' }}
        }},
        scales: {{
          x: {{ type: xType, ticks: {{ color:'#8b949e', maxTicksLimit:10 }},
               grid: {{ color:'#21262d' }} }},
          y: {{ ticks: {{ color:'#8b949e' }}, grid: {{ color:'#21262d' }},
               title: {{ display:true, text:yLabel, color:'#8b949e' }} }}
        }}
      }}
    }});
  }}

  const pnlData = {json.dumps(pnl_datasets)};
  makeChart('pnlChart', pnlData, 'Portfolio Value (USDT)');

  const predData = {json.dumps(pred_datasets)};
  makeChart('predChart', predData, 'Predicted Rate (%)');
  </script>

  <p style="color:#30363d;font-size:.75rem;text-align:center;margin-top:40px">
    Generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M')} UTC &nbsp;|&nbsp;
    HL Carry System &nbsp;|&nbsp; Paper Trading Mode
  </p>
</body>
</html>"""

    DASHBOARD.write_text(html, encoding="utf-8")
    logger.info("Dashboard written → %s", DASHBOARD)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_dashboard()
    print(f"Dashboard saved to {DASHBOARD}")
