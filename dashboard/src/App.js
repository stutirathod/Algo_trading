import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ReferenceLine,
  ComposedChart
} from "recharts";

// ─── API base — change this if Flask runs on a different port ─────────────────
const API = "http://localhost:5000/api";

// ─── Colour tokens ────────────────────────────────────────────────────────────
const C = {
  bg: "#080c14", surface: "#0d1421", card: "#111827", border: "#1e2d42",
  accent: "#00d4ff", accentDim: "#0099bb", green: "#00e8a2",
  red: "#ff4d6d", amber: "#ffb547", purple: "#a78bfa",
  text: "#e2e8f0", textMuted: "#64748b", textDim: "#334155",
};

// ─── Formatters ───────────────────────────────────────────────────────────────
const fmt = {
  pct:   v => v != null ? `${(v * 100).toFixed(2)}%` : "—",
  pct1:  v => v != null ? `${(v * 100).toFixed(1)}%`  : "—",
  num4:  v => v != null ? Number(v).toFixed(4) : "—",
  num5:  v => v != null ? Number(v).toFixed(5) : "—",
  price: v => v != null ? `₹${Number(v).toLocaleString("en-IN", { maximumFractionDigits: 2 })}` : "—",
};

// ─── useFetch hook ────────────────────────────────────────────────────────────
function useFetch(url, deps = []) {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    if (!url) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(url)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(d => { if (!cancelled) { setData(d); setLoading(false); } })
      .catch(e => { if (!cancelled) { setError(e.message); setLoading(false); } });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, ...deps]);

  return { data, loading, error };
}

// ─── Shared UI pieces ─────────────────────────────────────────────────────────
function Spinner() {
  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center",
      padding: 40, color: C.textMuted, fontSize: 13 }}>
      <div style={{ width: 18, height: 18, borderRadius: "50%",
        border: `2px solid ${C.border}`, borderTopColor: C.accent,
        animation: "spin 0.8s linear infinite", marginRight: 10 }} />
      Loading…
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

function ErrorBox({ msg }) {
  return (
    <div style={{ background: `${C.red}10`, border: `1px solid ${C.red}30`,
      borderRadius: 8, padding: "12px 16px", color: C.red, fontSize: 12,
      fontFamily: "monospace" }}>
      ⚠ {msg}
      <div style={{ marginTop: 6, color: C.textMuted }}>
        Make sure Flask is running: <code>python app.py</code>
      </div>
    </div>
  );
}

function SectionTitle({ children, accent }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
      <div style={{ width: 3, height: 18, background: accent || C.accent, borderRadius: 2 }} />
      <span style={{ fontSize: 13, fontWeight: 600, color: C.text, letterSpacing: "0.05em",
        textTransform: "uppercase", fontFamily: "monospace" }}>{children}</span>
    </div>
  );
}

const CustomTooltip = ({ active, payload, label, fmtFn }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#0a0f1a", border: `1px solid ${C.border}`, borderRadius: 8,
      padding: "10px 14px", fontSize: 12, fontFamily: "monospace" }}>
      <div style={{ color: C.textMuted, marginBottom: 6 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || C.accent, display: "flex",
          gap: 10, justifyContent: "space-between" }}>
          <span>{p.name}</span>
          <span style={{ fontWeight: 600 }}>
            {fmtFn ? fmtFn(p.value) : p.value}
          </span>
        </div>
      ))}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// TAB: OVERVIEW
// ─────────────────────────────────────────────────────────────────────────────
function OverviewTab({ ticker, signals }) {
  const { data: prices, loading: pL, error: pE } = useFetch(`${API}/price/${ticker}`, [ticker]);
  const { data: metrics, loading: mL, error: mE } = useFetch(`${API}/metrics/${ticker}`, [ticker]);
  const { data: bt }                               = useFetch(`${API}/backtest/${ticker}`, [ticker]);

  if (pL || mL) return <Spinner />;
  if (pE)       return <ErrorBox msg={pE} />;

  const last252 = (prices || []).slice(-252);
  const sig     = (signals || []).find(s => s.ticker === ticker);
  const latest  = last252[last252.length - 1] || {};
  const prev    = last252[last252.length - 2] || {};
  const dayChg  = prev.Close ? (latest.Close / prev.Close - 1) : 0;
  const btM     = bt?.metrics || {};

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Signal banner */}
      {sig && (
        <div style={{
          background: sig.direction === "BUY" ? `${C.green}10` : `${C.red}10`,
          border: `1px solid ${sig.direction === "BUY" ? C.green : C.red}40`,
          borderRadius: 12, padding: "18px 24px",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{
              width: 48, height: 48, borderRadius: "50%", fontSize: 22,
              background: sig.direction === "BUY" ? `${C.green}20` : `${C.red}20`,
              border: `2px solid ${sig.direction === "BUY" ? C.green : C.red}`,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>{sig.direction === "BUY" ? "↑" : "↓"}</div>
            <div>
              <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "monospace",
                color: sig.direction === "BUY" ? C.green : C.red }}>
                {sig.direction} SIGNAL
              </div>
              <div style={{ fontSize: 12, color: C.textMuted, marginTop: 2 }}>
                Ensemble: {fmt.num5(sig.signal)} · Conviction: {fmt.num5(sig.conviction)}
              </div>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 28, fontWeight: 700, fontFamily: "'DM Mono', monospace",
              color: C.text }}>{fmt.price(sig.current_price ?? sig.price)}</div>
            <div style={{ fontSize: 13, color: dayChg >= 0 ? C.green : C.red }}>
              {dayChg >= 0 ? "▲" : "▼"} {fmt.pct1(Math.abs(dayChg))} today
            </div>
          </div>
        </div>
      )}

      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        {[
          { label: "CAGR",           val: btM.cagr,          color: C.green, f: fmt.pct1 },
          { label: "Sharpe Ratio",   val: btM.sharpe_ratio,  color: C.accent, f: v => Number(v).toFixed(2) },
          { label: "Max Drawdown",   val: btM.max_drawdown,  color: C.red,   f: fmt.pct1 },
          { label: "Win Rate",       val: btM.win_rate,      color: C.amber, f: fmt.pct1 },
        ].map(k => (
          <div key={k.label} style={{ background: C.card, border: `1px solid ${C.border}`,
            borderRadius: 10, padding: "14px 18px", position: "relative", overflow: "hidden" }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2,
              background: `linear-gradient(90deg, ${k.color}44, ${k.color}, ${k.color}44)` }} />
            <div style={{ fontSize: 11, color: C.textMuted, textTransform: "uppercase",
              letterSpacing: "0.1em", fontFamily: "monospace" }}>{k.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: k.color,
              fontFamily: "'DM Mono', monospace", marginTop: 4 }}>
              {k.val != null ? k.f(k.val) : <span style={{ color: C.textDim }}>Run pipeline</span>}
            </div>
          </div>
        ))}
      </div>

      {/* Price chart */}
      <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
        <SectionTitle>Price + SMA + Bollinger Bands (1Y)</SectionTitle>
        {last252.length === 0 ? <ErrorBox msg="No price data" /> : (
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={last252} margin={{ top: 5, right: 10, bottom: 0, left: 10 }}>
              <defs>
                <linearGradient id="pg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={C.accent} stopOpacity={0.15} />
                  <stop offset="95%" stopColor={C.accent} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="date" tick={{ fill: C.textMuted, fontSize: 10 }}
                tickFormatter={v => v?.slice(5)} interval={30} />
              <YAxis tick={{ fill: C.textMuted, fontSize: 10 }} width={65}
                tickFormatter={v => `₹${(v / 1000).toFixed(1)}k`} />
              <Tooltip content={<CustomTooltip fmtFn={v => `₹${Number(v).toFixed(2)}`} />} />
              <Area  type="monotone" dataKey="Close"    stroke={C.accent}  strokeWidth={2} fill="url(#pg)"  name="Close"  dot={false} />
              <Line  type="monotone" dataKey="sma_20"   stroke={C.amber}   strokeWidth={1} dot={false} name="SMA20"  strokeDasharray="4 2" />
              <Line  type="monotone" dataKey="bb_upper" stroke={C.purple}  strokeWidth={1} dot={false} name="BB Hi"  strokeDasharray="2 3" opacity={0.6} />
              <Line  type="monotone" dataKey="bb_lower" stroke={C.purple}  strokeWidth={1} dot={false} name="BB Lo"  strokeDasharray="2 3" opacity={0.6} />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* RSI + MACD */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        {[
          { title: "RSI (14)", key: "rsi", color: C.amber, fmt: v => v?.toFixed(1),
            refs: [{ y: 70, c: C.red }, { y: 30, c: C.green }] },
          { title: "MACD Histogram", key: "macd_diff", color: C.purple, fmt: v => v?.toFixed(4) },
        ].map(panel => (
          <div key={panel.title} style={{ background: C.card, border: `1px solid ${C.border}`,
            borderRadius: 12, padding: 18 }}>
            <SectionTitle accent={panel.color}>{panel.title}</SectionTitle>
            <ResponsiveContainer width="100%" height={130}>
              <ComposedChart data={last252.slice(-90)} margin={{ top: 5, right: 5, bottom: 0, left: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="date" tick={{ fill: C.textMuted, fontSize: 9 }}
                  tickFormatter={v => v?.slice(5)} interval={20} />
                <YAxis tick={{ fill: C.textMuted, fontSize: 9 }} width={38} />
                <Tooltip content={<CustomTooltip fmtFn={panel.fmt} />} />
                {(panel.refs || []).map(r => (
                  <ReferenceLine key={r.y} y={r.y} stroke={r.c} strokeDasharray="3 3" strokeOpacity={0.6} />
                ))}
                <Line type="monotone" dataKey={panel.key} stroke={panel.color}
                  strokeWidth={1.5} dot={false} name={panel.title} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>

      {/* Model cards */}
      {mE ? <ErrorBox msg={mE} /> : metrics && (
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
          <SectionTitle>Model Performance — Test Set</SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
            {Object.entries(metrics).map(([name, d]) => {
              const isEns = name === "ensemble";
              return (
                <div key={name} style={{
                  background: isEns ? `${C.accent}10` : C.surface,
                  border: `1px solid ${isEns ? C.accent : C.border}`,
                  borderRadius: 10, padding: "14px 16px",
                }}>
                  <div style={{ fontSize: 11, color: C.textMuted, fontFamily: "monospace",
                    textTransform: "uppercase", marginBottom: 8 }}>
                    {name === "randomforest" ? "Rand. Forest" : name.toUpperCase()}
                  </div>
                  <div style={{ fontSize: 24, fontWeight: 700, fontFamily: "'DM Mono', monospace",
                    color: isEns ? C.accent : C.text }}>{fmt.pct1(d.dir_acc)}</div>
                  <div style={{ marginTop: 8, fontSize: 11, display: "flex",
                    flexDirection: "column", gap: 3 }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ color: C.textMuted }}>RMSE</span>
                      <span style={{ color: C.text, fontFamily: "monospace" }}>{fmt.num5(d.rmse)}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ color: C.textMuted }}>R²</span>
                      <span style={{ color: C.text, fontFamily: "monospace" }}>{fmt.num4(d.r2)}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB: BACKTEST
// ─────────────────────────────────────────────────────────────────────────────
function BacktestTab({ ticker }) {
  const { data: bt, loading, error } = useFetch(`${API}/backtest/${ticker}`, [ticker]);
  if (loading) return <Spinner />;
  if (error)   return <ErrorBox msg={error} />;

  const m      = bt?.metrics || {};
  const equity = bt?.equity_curve || [];

  const withDD = equity.map((d, i) => {
    const peak = Math.max(...equity.slice(0, i + 1).map(x => x.portfolio_value || 0));
    return { ...d, drawdown: peak > 0 ? ((d.portfolio_value - peak) / peak * 100) : 0 };
  });

  const metaCards = [
    { label: "Total Return",  val: m.total_return, f: fmt.pct,  color: C.green  },
    { label: "CAGR",          val: m.cagr,         f: fmt.pct1, color: C.green  },
    { label: "Sharpe Ratio",  val: m.sharpe_ratio, f: v => Number(v).toFixed(2), color: C.accent },
    { label: "Sortino Ratio", val: m.sortino_ratio,f: v => Number(v).toFixed(2), color: C.accent },
    { label: "Max Drawdown",  val: m.max_drawdown, f: fmt.pct1, color: C.red    },
    { label: "Calmar Ratio",  val: m.calmar_ratio, f: v => Number(v).toFixed(2), color: C.amber  },
    { label: "Win Rate",      val: m.win_rate,     f: fmt.pct1, color: C.amber  },
    { label: "Profit Factor", val: m.profit_factor,f: v => Number(v).toFixed(2), color: C.purple },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        {metaCards.map(k => (
          <div key={k.label} style={{ background: C.card, border: `1px solid ${C.border}`,
            borderRadius: 10, padding: "16px 18px", position: "relative", overflow: "hidden" }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2,
              background: `linear-gradient(90deg, transparent, ${k.color}, transparent)` }} />
            <div style={{ fontSize: 11, color: C.textMuted, textTransform: "uppercase",
              letterSpacing: "0.08em", fontFamily: "monospace", marginBottom: 6 }}>{k.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: k.color,
              fontFamily: "'DM Mono', monospace" }}>
              {k.val != null ? k.f(k.val) : <span style={{ color: C.textDim, fontSize: 14 }}>—</span>}
            </div>
          </div>
        ))}
      </div>

      {equity.length > 0 ? (
        <>
          <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
            <SectionTitle>Portfolio Equity Curve</SectionTitle>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={withDD} margin={{ top: 5, right: 10, bottom: 0, left: 10 }}>
                <defs>
                  <linearGradient id="eg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={C.green} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={C.green} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="date" tick={{ fill: C.textMuted, fontSize: 10 }}
                  tickFormatter={v => v?.slice(0, 7)} interval={40} />
                <YAxis tick={{ fill: C.textMuted, fontSize: 10 }} width={80}
                  tickFormatter={v => `₹${(v / 1e5).toFixed(1)}L`} />
                <Tooltip content={<CustomTooltip fmtFn={v => `₹${(v / 1e5).toFixed(2)}L`} />} />
                <Area type="monotone" dataKey="portfolio_value" stroke={C.green} strokeWidth={2}
                  fill="url(#eg)" name="Portfolio" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
            <SectionTitle accent={C.red}>Drawdown</SectionTitle>
            <ResponsiveContainer width="100%" height={130}>
              <AreaChart data={withDD} margin={{ top: 5, right: 10, bottom: 0, left: 10 }}>
                <defs>
                  <linearGradient id="dg" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={C.red} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={C.red} stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="date" tick={{ fill: C.textMuted, fontSize: 10 }}
                  tickFormatter={v => v?.slice(0, 7)} interval={40} />
                <YAxis tick={{ fill: C.textMuted, fontSize: 10 }} width={45}
                  tickFormatter={v => `${v.toFixed(0)}%`} />
                <Tooltip content={<CustomTooltip fmtFn={v => `${v.toFixed(2)}%`} />} />
                <Area type="monotone" dataKey="drawdown" stroke={C.red} strokeWidth={1.5}
                  fill="url(#dg)" name="Drawdown" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </>
      ) : (
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12,
          padding: 40, textAlign: "center", color: C.textMuted, fontSize: 13 }}>
          No equity curve found. Run the pipeline first to generate backtest results.
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB: FEATURES
// ─────────────────────────────────────────────────────────────────────────────
function FeaturesTab({ ticker }) {
  const { data: feats, loading, error } = useFetch(`${API}/features/${ticker}`, [ticker]);
  const { data: eda,   loading: eL }    = useFetch(`${API}/eda/${ticker}`,      [ticker]);

  if (loading || eL) return <Spinner />;
  if (error) return <ErrorBox msg={error} />;

  const top20 = (feats || []).slice(0, 20);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* EDA stats row */}
      {eda && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
          {[
            { label: "Ann. Volatility",  val: fmt.pct1(eda.ann_volatility),  color: C.amber  },
            { label: "Skewness",         val: Number(eda.skewness).toFixed(3),  color: C.purple },
            { label: "Excess Kurtosis",  val: Number(eda.excess_kurtosis).toFixed(2), color: C.red },
            { label: "Normal Returns?",  val: eda.is_normal ? "YES" : "NO",   color: eda.is_normal ? C.green : C.red },
          ].map(k => (
            <div key={k.label} style={{ background: C.card, border: `1px solid ${C.border}`,
              borderRadius: 10, padding: "14px 18px" }}>
              <div style={{ fontSize: 11, color: C.textMuted, fontFamily: "monospace",
                textTransform: "uppercase", marginBottom: 4 }}>{k.label}</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: k.color,
                fontFamily: "'DM Mono', monospace" }}>{k.val}</div>
            </div>
          ))}
        </div>
      )}

      {/* Return distribution */}
      {eda?.histogram && (
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
          <SectionTitle>Return Distribution (EDA)</SectionTitle>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={eda.histogram} margin={{ top: 5, right: 10, bottom: 0, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="x" tick={{ fill: C.textMuted, fontSize: 9 }}
                tickFormatter={v => `${(v * 100).toFixed(1)}%`} interval={10} />
              <YAxis tick={{ fill: C.textMuted, fontSize: 9 }} width={35} />
              <Tooltip content={<CustomTooltip fmtFn={v => v} />} />
              <Bar dataKey="count" fill={C.accent} opacity={0.8} name="Freq" radius={[2,2,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Feature importances */}
      {top20.length > 0 ? (
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
          <SectionTitle>Feature Importances — RF vs XGBoost (from trained models)</SectionTitle>
          <ResponsiveContainer width="100%" height={Math.max(240, top20.length * 22)}>
            <BarChart data={top20} layout="vertical"
              margin={{ top: 5, right: 20, bottom: 5, left: 110 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} horizontal={false} />
              <XAxis type="number" tick={{ fill: C.textMuted, fontSize: 10 }}
                tickFormatter={v => `${(v * 100).toFixed(1)}%`} />
              <YAxis type="category" dataKey="feature"
                tick={{ fill: C.text, fontSize: 10, fontFamily: "monospace" }} width={110} />
              <Tooltip content={<CustomTooltip fmtFn={v => `${(v * 100).toFixed(3)}%`} />} />
              <Bar dataKey="rf"  fill={C.amber} name="Random Forest" radius={[0,3,3,0]} opacity={0.85} />
              <Bar dataKey="xgb" fill={C.accent} name="XGBoost"      radius={[0,3,3,0]} opacity={0.85} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", gap: 16, marginTop: 8 }}>
            {[["Random Forest", C.amber], ["XGBoost", C.accent]].map(([n, c]) => (
              <div key={n} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: C.textMuted }}>
                <div style={{ width: 10, height: 10, background: c, borderRadius: 2 }} />
                {n}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12,
          padding: 40, textAlign: "center", color: C.textMuted, fontSize: 13 }}>
          Feature importances will appear after training. Run the pipeline first.
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB: SIGNALS
// ─────────────────────────────────────────────────────────────────────────────
function SignalsTab({ signals, sigLoading, sigError, onRefresh }) {
  if (sigLoading) return <Spinner />;
  if (sigError)   return <ErrorBox msg={sigError} />;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontSize: 13, color: C.textMuted }}>
          Last updated: <span style={{ color: C.text }}>{new Date().toLocaleString("en-IN")}</span>
        </div>
        <button onClick={onRefresh} style={{
          padding: "6px 14px", borderRadius: 7, fontSize: 12, cursor: "pointer",
          background: `${C.accent}15`, border: `1px solid ${C.accent}40`,
          color: C.accent, fontFamily: "monospace",
        }}>⟳ Refresh Signals</button>
      </div>

      {(!signals || signals.length === 0) ? (
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12,
          padding: 40, textAlign: "center", color: C.textMuted }}>
          No signals found. Run: <code style={{ color: C.accent }}>python pipeline.py --mode predict --tickers RELIANCE.NS TCS.NS</code>
        </div>
      ) : (
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
          <SectionTitle>Today's Trading Signals</SectionTitle>
          {signals.map(s => {
            const dir = s.direction === "BUY";
            return (
              <div key={s.ticker} style={{
                background: dir ? `${C.green}08` : `${C.red}08`,
                border: `1px solid ${dir ? C.green : C.red}30`,
                borderRadius: 10, padding: "16px 20px", marginBottom: 10,
                display: "grid", gridTemplateColumns: "200px 80px 1fr 1fr 1fr auto",
                alignItems: "center", gap: 16,
              }}>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: C.text, fontFamily: "monospace" }}>
                    {s.ticker}
                  </div>
                  <div style={{ fontSize: 11, color: C.textMuted }}>
                    {fmt.price(s.current_price ?? s.price)}
                  </div>
                </div>
                <div style={{
                  padding: "4px 12px", borderRadius: 6, fontSize: 12, fontWeight: 700,
                  background: dir ? `${C.green}20` : `${C.red}20`,
                  color: dir ? C.green : C.red, textAlign: "center", fontFamily: "monospace",
                }}>{s.direction}</div>
                {[
                  { label: "Ensemble", val: s.signal,     color: C.accent },
                  { label: "RF",       val: s.rf,         color: C.amber  },
                  { label: "XGBoost",  val: s.xgb,        color: C.purple },
                ].map(col => (
                  <div key={col.label} style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 10, color: C.textMuted, fontFamily: "monospace",
                      textTransform: "uppercase", marginBottom: 2 }}>{col.label}</div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: col.color,
                      fontFamily: "monospace" }}>
                      {col.val != null ? (col.val > 0 ? "+" : "") + fmt.num5(col.val) : "—"}
                    </div>
                  </div>
                ))}
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 10, color: C.textMuted, marginBottom: 3 }}>Conviction</div>
                  <div style={{ display: "flex", gap: 3, justifyContent: "flex-end" }}>
                    {[...Array(5)].map((_, i) => (
                      <div key={i} style={{
                        width: 6, height: 16, borderRadius: 2,
                        background: i < Math.ceil((s.conviction / 0.005) * 5)
                          ? (dir ? C.green : C.red) : C.border,
                      }} />
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TAB: PIPELINE CONTROL
// ─────────────────────────────────────────────────────────────────────────────
function PipelineTab({ allTickers }) {
  const [selectedTickers, setSelectedTickers] = useState(allTickers.slice(0, 3));
  const [useLSTM,    setUseLSTM]    = useState(false);
  const [status,     setStatus]     = useState(null);
  const logRef = useRef(null);

  // Poll status while running
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const r = await fetch(`${API}/pipeline/status`);
        const d = await r.json();
        setStatus(d);
        if (logRef.current) {
          logRef.current.scrollTop = logRef.current.scrollHeight;
        }
      } catch {}
    }, 1500);
    return () => clearInterval(poll);
  }, []);

  const startTraining = async () => {
    await fetch(`${API}/pipeline/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tickers: selectedTickers, lstm: useLSTM }),
    });
  };

  const runPredict = async () => {
    const r = await fetch(`${API}/pipeline/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tickers: selectedTickers }),
    });
    const d = await r.json();
    alert(d.status === "ok" ? "Signals generated! Switch to Signals tab." : `Error: ${d.stderr}`);
  };

  const knownTickers = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","KOTAKBANK.NS","HINDUNILVR.NS","BHARTIARTL.NS","ITC.NS",
  ];

  const toggleTicker = t =>
    setSelectedTickers(p => p.includes(t) ? p.filter(x => x !== t) : [...p, t]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Controls */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
          <SectionTitle>Select Tickers</SectionTitle>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {knownTickers.map(t => {
              const sel = selectedTickers.includes(t);
              return (
                <button key={t} onClick={() => toggleTicker(t)} style={{
                  padding: "5px 12px", borderRadius: 6, fontSize: 11, cursor: "pointer",
                  fontFamily: "monospace", border: "none", transition: "all 0.15s",
                  background: sel ? `${C.accent}20` : C.surface,
                  color: sel ? C.accent : C.textMuted,
                  outline: sel ? `1px solid ${C.accent}50` : `1px solid ${C.border}`,
                }}>{t.replace(".NS", "")}</button>
              );
            })}
          </div>
          <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 10 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer",
              fontSize: 12, color: C.textMuted }}>
              <input type="checkbox" checked={useLSTM}
                onChange={e => setUseLSTM(e.target.checked)}
                style={{ accentColor: C.accent, width: 14, height: 14 }} />
              Include LSTM (requires TensorFlow, slower)
            </label>
          </div>
        </div>

        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
          <SectionTitle>Run Pipeline</SectionTitle>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <button onClick={startTraining}
              disabled={status?.running || selectedTickers.length === 0}
              style={{
                padding: "12px 20px", borderRadius: 8, fontSize: 13, cursor: "pointer",
                fontFamily: "monospace", border: "none", fontWeight: 600,
                background: status?.running ? C.border : `linear-gradient(135deg, ${C.accent}, ${C.accentDim})`,
                color: status?.running ? C.textMuted : "#000",
                opacity: selectedTickers.length === 0 ? 0.5 : 1,
                transition: "all 0.2s",
              }}>
              {status?.running ? "⏳ Training in progress…" : "▶ Start Full Training Pipeline"}
            </button>
            <button onClick={runPredict}
              disabled={status?.running}
              style={{
                padding: "10px 20px", borderRadius: 8, fontSize: 13, cursor: "pointer",
                fontFamily: "monospace", border: `1px solid ${C.green}50`, fontWeight: 500,
                background: `${C.green}10`, color: C.green, transition: "all 0.2s",
              }}>
              ⚡ Generate Today's Signals (predict mode)
            </button>
          </div>

          {/* Pipeline steps */}
          {status && (
            <div style={{ marginTop: 14 }}>
              <div style={{ height: 4, background: C.border, borderRadius: 2, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${status.progress}%`,
                  background: `linear-gradient(90deg, ${C.accent}, ${C.green})`,
                  borderRadius: 2, transition: "width 0.5s ease" }} />
              </div>
              <div style={{ fontSize: 11, color: C.textMuted, marginTop: 4, fontFamily: "monospace" }}>
                {status.running ? `Training… ${status.progress}%` : `Last run: ${status.started_at || "—"}`}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Live log */}
      <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
          marginBottom: 12 }}>
          <SectionTitle>Live Pipeline Log</SectionTitle>
          {status?.running && (
            <div style={{ display: "flex", alignItems: "center", gap: 6,
              padding: "3px 10px", background: `${C.green}15`,
              border: `1px solid ${C.green}30`, borderRadius: 6 }}>
              <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.green,
                animation: "pulse 1s infinite" }} />
              <span style={{ fontSize: 11, color: C.green, fontFamily: "monospace" }}>RUNNING</span>
            </div>
          )}
        </div>
        <div ref={logRef} style={{
          background: "#050810", border: `1px solid ${C.border}`, borderRadius: 8,
          padding: "12px 14px", height: 320, overflowY: "auto",
          fontFamily: "monospace", fontSize: 11, lineHeight: 1.6,
        }}>
          {!status?.log?.length ? (
            <div style={{ color: C.textMuted }}>
              Pipeline log will appear here when you start training.
            </div>
          ) : (
            status.log.map((line, i) => (
              <div key={i} style={{
                color: line.includes("ERROR") ? C.red
                     : line.includes("COMPLETE") || line.includes("STEP") ? C.accent
                     : line.includes("WARNING") ? C.amber
                     : C.textMuted,
              }}>{line}</div>
            ))
          )}
        </div>
      </div>

      {/* Architecture diagram */}
      <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
        <SectionTitle>System Architecture</SectionTitle>
        <div style={{ display: "flex", alignItems: "center", gap: 0, overflowX: "auto",
          padding: "10px 0" }}>
          {[
            { step: "1", label: "Data\nCollection", color: C.purple,  desc: "NSE/BSE OHLCV\n+ Fundamentals" },
            { step: "2", label: "EDA",              color: C.accent,  desc: "Stationarity\nARCH Tests" },
            { step: "3", label: "Features",         color: C.amber,   desc: "80+ Technical\n& Fundamental" },
            { step: "4", label: "LSTM",             color: C.green,   desc: "Sequential\nPatterns" },
            { step: "5", label: "RF + XGB",         color: C.green,   desc: "Feature\nInteractions" },
            { step: "6", label: "Ensemble",         color: C.accent,  desc: "OOF Stacking\nRidge Meta" },
            { step: "7", label: "Backtest",         color: C.amber,   desc: "Sharpe/DD\nCAGR" },
            { step: "8", label: "Risk\nMgmt",       color: C.red,     desc: "Stop-Loss\nSizing" },
          ].map((s, i, arr) => (
            <div key={s.step} style={{ display: "flex", alignItems: "center", flexShrink: 0 }}>
              <div style={{ textAlign: "center", width: 90 }}>
                <div style={{ width: 36, height: 36, borderRadius: "50%",
                  background: `${s.color}20`, border: `2px solid ${s.color}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 13, fontWeight: 700, color: s.color,
                  margin: "0 auto 6px", fontFamily: "monospace" }}>{s.step}</div>
                <div style={{ fontSize: 11, color: C.text, fontWeight: 600,
                  whiteSpace: "pre-line", lineHeight: 1.3 }}>{s.label}</div>
                <div style={{ fontSize: 9, color: C.textMuted, marginTop: 3,
                  whiteSpace: "pre-line", lineHeight: 1.3 }}>{s.desc}</div>
              </div>
              {i < arr.length - 1 && (
                <div style={{ width: 20, height: 1, background: C.border,
                  flexShrink: 0, margin: "0 2px" }} />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [ticker, setTicker]   = useState("RELIANCE.NS");
  const [tab,    setTab]      = useState("overview");
  const [time,   setTime]     = useState(new Date());
  const [sigKey, setSigKey]   = useState(0);   // increment to refresh signals

  const { data: tickers } = useFetch(`${API}/tickers`);
  const allTickers = tickers || ["RELIANCE.NS", "TCS.NS", "INFY.NS"];

  const { data: signals, loading: sigL, error: sigE } =
    useFetch(`${API}/signals`, [sigKey]);

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // Set default ticker once tickers load
  useEffect(() => {
    if (allTickers.length > 0 && !allTickers.includes(ticker)) {
      setTicker(allTickers[0]);
    }
  }, [allTickers]);

  const tabs = [
    { id: "overview",  label: "Overview",  icon: "◈" },
    { id: "backtest",  label: "Backtest",  icon: "⟳" },
    { id: "features",  label: "Features",  icon: "◉" },
    { id: "signals",   label: "Signals",   icon: "▲" },
    { id: "pipeline",  label: "Pipeline",  icon: "⬡" },
  ];

  const sigForTicker = (signals || []).find(s => s.ticker === ticker);

  return (
    <div style={{ background: C.bg, minHeight: "100vh",
      fontFamily: "'IBM Plex Sans', sans-serif", color: C.text, fontSize: 14 }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
      `}</style>

      {/* Top bar */}
      <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`,
        padding: "0 24px", display: "flex", alignItems: "center",
        justifyContent: "space-between", height: 52, position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 28, height: 28, borderRadius: 6, fontSize: 14, fontWeight: 700,
              background: `linear-gradient(135deg, ${C.accent}, ${C.purple})`,
              display: "flex", alignItems: "center", justifyContent: "center", color: "#000" }}>A</div>
            <span style={{ fontSize: 14, fontWeight: 600, letterSpacing: "0.03em" }}>AlgoTrader</span>
            <span style={{ fontSize: 11, color: C.textMuted, fontFamily: "monospace",
              background: C.border, padding: "2px 6px", borderRadius: 4 }}>NSE/BSE</span>
          </div>
          {/* Ticker pills */}
          <div style={{ display: "flex", gap: 4, marginLeft: 8 }}>
            {allTickers.map(t => {
              const s = (signals || []).find(x => x.ticker === t);
              return (
                <button key={t} onClick={() => setTicker(t)} style={{
                  padding: "4px 12px", borderRadius: 6, fontSize: 12, cursor: "pointer",
                  fontFamily: "monospace", border: "none", transition: "all 0.2s",
                  background: ticker === t ? `${C.accent}20` : "transparent",
                  color: ticker === t ? C.accent : C.textMuted,
                  outline: ticker === t ? `1px solid ${C.accent}50` : "none",
                }}>
                  {t.replace(".NS","").replace(".BO","")}
                  {s && <span style={{ marginLeft: 4, fontSize: 10,
                    color: s.direction === "BUY" ? C.green : C.red }}>
                    {s.direction === "BUY" ? "▲" : "▼"}
                  </span>}
                </button>
              );
            })}
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ fontSize: 11, color: C.textMuted, fontFamily: "monospace" }}>
            {time.toLocaleTimeString("en-IN")} IST
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6,
            padding: "4px 10px", background: `${C.green}15`,
            border: `1px solid ${C.green}30`, borderRadius: 6 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.green,
              animation: "pulse 2s infinite" }} />
            <span style={{ fontSize: 11, color: C.green, fontFamily: "monospace" }}>
              API Connected
            </span>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", height: "calc(100vh - 52px)" }}>
        {/* Side nav */}
        <div style={{ width: 52, background: C.surface, borderRight: `1px solid ${C.border}`,
          display: "flex", flexDirection: "column", alignItems: "center",
          paddingTop: 16, gap: 4, flexShrink: 0 }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} title={t.label} style={{
              width: 36, height: 36, borderRadius: 8, border: "none", cursor: "pointer",
              background: tab === t.id ? `${C.accent}20` : "transparent",
              color: tab === t.id ? C.accent : C.textMuted,
              fontSize: 16, display: "flex", alignItems: "center", justifyContent: "center",
              outline: tab === t.id ? `1px solid ${C.accent}40` : "none",
              transition: "all 0.2s",
            }}>{t.icon}</button>
          ))}
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflow: "auto", padding: 20 }}>
          {/* Tab label row */}
          <div style={{ display: "flex", gap: 6, marginBottom: 20, alignItems: "center" }}>
            {tabs.map(t => (
              <button key={t.id} onClick={() => setTab(t.id)} style={{
                padding: "6px 14px", borderRadius: 7, fontSize: 12, cursor: "pointer",
                fontFamily: "monospace", border: "none", transition: "all 0.2s",
                background: tab === t.id ? `${C.accent}20` : C.surface,
                color: tab === t.id ? C.accent : C.textMuted,
                outline: tab === t.id ? `1px solid ${C.accent}40` : `1px solid ${C.border}`,
              }}>{t.icon} {t.label}</button>
            ))}
            <div style={{ marginLeft: "auto", fontSize: 12, color: C.textMuted,
              fontFamily: "monospace" }}>
              {tab !== "pipeline" && tab !== "signals" &&
                <>Viewing: <span style={{ color: C.accent, marginLeft: 4, fontWeight: 600 }}>{ticker}</span></>
              }
            </div>
          </div>

          {tab === "overview" && <OverviewTab ticker={ticker} signals={signals} />}
          {tab === "backtest" && <BacktestTab ticker={ticker} />}
          {tab === "features" && <FeaturesTab ticker={ticker} />}
          {tab === "signals"  && (
            <SignalsTab signals={signals} sigLoading={sigL} sigError={sigE}
              onRefresh={() => setSigKey(k => k + 1)} />
          )}
          {tab === "pipeline" && <PipelineTab allTickers={allTickers} />}
        </div>
      </div>
    </div>
  );
}
