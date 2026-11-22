"""
app.py
======
PitchSense AI - Professional dark-theme Streamlit dashboard for the Football Tactical Expert System.

Run with:   python -m streamlit run app.py
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from expert_system import (
    CSV_COLUMNS, DEFAULT_FACTS, DX_CENTRAL, DX_DISJOINT, DX_EXHAUSTION, DX_INJURY, DX_PASSIVE,
    DX_PREFIX, DX_PRESS_RESIST, DX_PRESSING, DX_THREAT, FACT_COLUMNS, FACT_SPECS, POSITIONS, RULES,
    InferenceEngine, analyze_timeline, cast_fact, row_to_facts, sample_csv_bytes, sample_dataframe,
    validate_dataframe,
)

st.set_page_config(page_title="PitchSense AI · Football Tactical Expert System", page_icon="⚽", layout="wide")

# ===========================================================================
# 1. THEME (Fixed CSS - preserves Streamlit Material Icons & prevents text wrap)
# ===========================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0B0F19;--card:#111827;--border:#1F2937;--emerald:#10B981;--green:#00C853;--text:#E5E7EB;--muted:#9CA3AF;}

/* Apply Inter font ONLY to text elements, avoiding Streamlit's Material Symbols icons */
.stApp, .stMarkdown, p, h1, h2, h3, h4, label, input, textarea, select, .kpi, .hero, .chip, .alert, .brand {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.stApp{background:radial-gradient(1100px 520px at 8% -8%, rgba(16,185,129,.14), transparent 60%),
       radial-gradient(900px 420px at 100% 0%, rgba(0,200,83,.06), transparent 60%), var(--bg);color:var(--text);}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.2rem;max-width:1600px;}
[data-testid="stSidebar"]{background:#0A101C;border-right:1px solid var(--border);}
h1,h2,h3,h4{color:#F9FAFB;letter-spacing:-.01em;}

.brand{display:flex;align-items:center;gap:12px;font-weight:800;font-size:1.3rem;color:#F9FAFB;margin-bottom:.6rem;letter-spacing:.02em;}
.brand .dot{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#00C853,#10B981);
            display:flex;align-items:center;justify-content:center;font-size:1.15rem;box-shadow:0 0 18px rgba(16,185,129,.45);}
.brand span.ai{color:#10B981;}

.hero{background:linear-gradient(135deg,#0F1F1A 0%,#111827 55%,#0B1220 100%);border:1px solid var(--border);
      border-radius:18px;padding:20px 26px;display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;
      box-shadow:0 10px 30px rgba(0,0,0,.35);}
.hero .eyebrow{font-size:.72rem;letter-spacing:.18em;color:var(--emerald);font-weight:700;text-transform:uppercase;}
.hero .title{font-size:1.85rem;font-weight:800;color:#F9FAFB;line-height:1.2;margin:3px 0;}
.hero .sub{color:var(--muted);font-size:.9rem;}
.pill{border:1px solid rgba(16,185,129,.5);background:rgba(16,185,129,.12);color:#6EE7B7;border-radius:999px;
      padding:6px 14px;font-size:.78rem;font-weight:700;white-space:nowrap;}

.chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px 0;}
.chip{background:var(--card);border:1px solid var(--border);border-radius:999px;padding:5px 12px;font-size:.78rem;color:var(--muted);}
.chip b{color:#F3F4F6;font-weight:600;}
.chip.bad b{color:#F87171;} .chip.warn b{color:#FBBF24;} .chip.ok b{color:#34D399;}

.kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px 14px;border-top:3px solid var(--emerald);
     box-shadow:0 4px 14px rgba(0,0,0,.25);height:100%;overflow:hidden;}
.kpi.warn{border-top-color:#F59E0B;} .kpi.bad{border-top-color:#EF4444;}
.kpi .label{font-size:.66rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600;white-space:nowrap;}
.kpi .value{font-size:1.45rem;font-weight:800;color:#F9FAFB;line-height:1.3;white-space:nowrap;margin:2px 0;}
.kpi.bad .value{color:#F87171;} .kpi.warn .value{color:#FBBF24;} .kpi.ok .value{color:#34D399;}
.kpi .sub{font-size:.72rem;color:var(--muted);white-space:nowrap;}

.section{font-size:.8rem;text-transform:uppercase;letter-spacing:.14em;color:var(--emerald);font-weight:700;margin:18px 0 8px 2px;}
[data-testid="stPlotlyChart"]{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:6px;}
.feed{max-height:560px;overflow-y:auto;padding-right:4px;}
.alert{background:var(--card);border:1px solid var(--border);border-left:5px solid var(--c);border-radius:12px;
       padding:12px 14px;margin-bottom:10px;box-shadow:0 4px 14px rgba(0,0,0,.25);}
.alert-top{display:flex;gap:8px;align-items:center;margin-bottom:4px;}
.badge{font-size:.64rem;font-weight:800;letter-spacing:.08em;padding:2px 8px;border-radius:6px;}
.rid{font-size:.72rem;color:var(--muted);font-weight:600;}
.tag{font-size:.66rem;color:#A7F3D0;background:rgba(16,185,129,.12);border-radius:6px;padding:2px 7px;}
.alert-title{font-weight:700;color:#F9FAFB;font-size:.98rem;}
.alert-rec{color:#D1D5DB;font-size:.86rem;margin-top:4px;}
.empty{background:var(--card);border:1px dashed #065F46;border-radius:12px;padding:22px;color:#6EE7B7;text-align:center;font-weight:600;}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.76rem;color:var(--muted);margin:6px 2px 0;}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px;}

.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{color:var(--muted);font-weight:600;}
.stTabs [aria-selected="true"]{color:var(--emerald)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--emerald)!important;}
[data-testid="stExpander"]{background:var(--card);border:1px solid var(--border);border-radius:12px;}
.stDownloadButton button,.stButton button{background:linear-gradient(135deg,#00C853,#10B981);color:#03180D;border:0;
      font-weight:700;border-radius:10px;}
.stDownloadButton button:hover,.stButton button:hover{filter:brightness(1.1);color:#03180D;}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# ===========================================================================
# 2. SMALL UI HELPERS
# ===========================================================================
SEV_COLOR = {"Critical": "#EF4444", "High": "#F59E0B", "Medium": "#3B82F6", "Info": "#10B981"}
SEV_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Info": 3}
esc = html.escape


def tone_hi(v: float, warn: float, bad: float) -> str:
    return "bad" if v > bad else "warn" if v > warn else "ok"


def tone_lo(v: float, warn: float, bad: float) -> str:
    return "bad" if v < bad else "warn" if v < warn else "ok"


def kpi(label: str, value: str, sub: str, tone: str) -> str:
    return (f'<div class="kpi {tone}"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(str(value))}</div><div class="sub">{esc(sub)}</div></div>')


def chip(label: str, value: str, tone: str = "") -> str:
    return f'<span class="chip {tone}">{esc(label)}: <b>{esc(str(value))}</b></span>'


def alert_card(f) -> str:
    c = SEV_COLOR[f.rule.severity]
    chained = any(cond.fact.startswith(DX_PREFIX) for cond, _ in f.evidence)
    tag = '<span class="tag">🔗 chained</span>' if chained else ""
    return (f'<div class="alert" style="--c:{c}"><div class="alert-top">'
            f'<span class="badge" style="background:{c}26;color:{c}">{f.rule.severity.upper()}</span>'
            f'<span class="rid">{f.rule.rule_id}</span>{tag}</div>'
            f'<div class="alert-title">{esc(f.rule.diagnosis)}</div>'
            f'<div class="alert-rec">➡ {esc(f.recommendation)}</div></div>')


# ===========================================================================
# 3. TACTICAL PITCH (Plotly)
# ===========================================================================
def build_pitch(wm: dict, fired: set[str]) -> go.Figure:
    LINE = "rgba(255,255,255,.85)"
    RED, AMBER, SKY, YELLOW, LIME = "#EF4444", "#FBBF24", "#38BDF8", "#FDE047", "#ECFCCB"

    mid_x = min(float(wm["Midfield_Line_Height_m"]), 92.0)
    gap = float(wm["Space_Between_Lines"])
    def_x = max(6.0, mid_x - gap)

    shapes: list[dict] = []
    anns: list[dict] = []
    fig = go.Figure()

    def rect(x0, x1, y0, y1, fill=None, line=None, width=2, dash=None):
        shapes.append(dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, layer="below",
                           fillcolor=fill or "rgba(0,0,0,0)",
                           line=dict(color=line or "rgba(0,0,0,0)", width=width if line else 0, dash=dash)))

    def glow(x0, x1, y0, y1, rgb="239,68,68", k=1.0):
        for pad, a in [(4, .05), (3, .07), (2, .10), (1, .14)]:
            rect(x0 - pad, x1 + pad, y0 - pad, y1 + pad, fill=f"rgba({rgb},{round(a * k, 3)})")
        rect(x0, x1, y0, y1, fill=f"rgba({rgb},.26)", line=f"rgba({rgb},.95)", width=2, dash="dot")

    def arrow(x0, y0, x1, y1, color, width=3):
        anns.append(dict(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                         arrowhead=3, arrowsize=1.3, arrowwidth=width, arrowcolor=color, text=""))

    def label(x, y, text, color, anchor="center"):
        anns.append(dict(x=x, y=y, text=text, showarrow=False, xanchor=anchor, font=dict(color=color, size=11),
                         bgcolor="rgba(11,15,25,.82)", bordercolor=color, borderwidth=1, borderpad=3))

    def line_trace(xs, ys, color=LINE, width=2, dash=None):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash),
                                 hoverinfo="skip", showlegend=False))

    for i in range(10):
        rect(i * 10.5, (i + 1) * 10.5, 0, 68, fill="#166534" if i % 2 == 0 else "#15803D")
    rect(0, 105, 0, 68, line=LINE)
    shapes.append(dict(type="line", x0=52.5, x1=52.5, y0=0, y1=68, line=dict(color=LINE, width=2), layer="below"))
    shapes.append(dict(type="circle", x0=52.5 - 9.15, x1=52.5 + 9.15, y0=34 - 9.15, y1=34 + 9.15,
                       line=dict(color=LINE, width=2), layer="below"))
    for x0, x1 in [(0, 16.5), (88.5, 105)]:
        rect(x0, x1, 13.84, 54.16, line=LINE)
    for x0, x1 in [(0, 5.5), (99.5, 105)]:
        rect(x0, x1, 24.84, 43.16, line=LINE)
    rect(-2, 0, 30.34, 37.66, line=LINE)
    rect(105, 107, 30.34, 37.66, line=LINE)
    t = np.linspace(-0.9273, 0.9273, 40)
    line_trace(11 + 9.15 * np.cos(t), 34 + 9.15 * np.sin(t))
    line_trace(94 - 9.15 * np.cos(t), 34 + 9.15 * np.sin(t))
    fig.add_trace(go.Scatter(x=[11, 94, 52.5], y=[34, 34, 34], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=5, color="white")))

    for x, col in [(def_x, SKY), (mid_x, YELLOW)]:
        shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=68, layer="below",
                           line=dict(color=col, width=2, dash="dash")))
    label(def_x, 72, f"DEF LINE · {def_x:.0f} m", SKY, anchor="right")
    label(mid_x, 72, f"MID LINE · {mid_x:.0f} m", YELLOW, anchor="left")
    gap_col = RED if gap > 15 else "#E5E7EB"
    anns.append(dict(x=mid_x, y=-3, ax=def_x, ay=-3, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                     arrowhead=2, arrowside="end+start", arrowwidth=2, arrowcolor=gap_col, text=""))
    label((def_x + mid_x) / 2, -6.2, f"gap {gap:.1f} m", gap_col)
    anns.append(dict(x=104, y=-6.2, text="Attacking direction ►", showarrow=False, xanchor="right",
                     font=dict(color="#9CA3AF", size=10)))
    anns.append(dict(x=0, y=-6.2, text="◄ OUR GOAL", showarrow=False, xanchor="left",
                     font=dict(color="#9CA3AF", size=10)))

    if DX_PASSIVE in fired:
        rect(0, mid_x, 0, 68, fill="rgba(251,146,60,.10)")
        label(mid_x / 2, 62, "PASSIVE DEEP BLOCK", "#FB923C")
    if DX_CENTRAL in fired:
        glow(def_x, mid_x, 18, 50)
        label((def_x + mid_x) / 2, 54, "CENTRAL CORRIDOR OPEN", RED)
        n = min(3, max(1, int(wm["Opponent_Through_Ball_Count"]) // 2))
        for y in [34, 26, 42][:n]:
            arrow(mid_x + 16, y + 6, def_x - 3, y, RED)
    if DX_DISJOINT in fired:
        glow(def_x, mid_x, 3, 65, rgb="239,68,68", k=.8)
        label((def_x + mid_x) / 2, 8, "EXPOSED BACKLINE", RED)
        step_x = min(def_x + 7, mid_x)
        shapes.append(dict(type="line", x0=step_x, x1=step_x, y0=0, y1=68, layer="below",
                           line=dict(color=AMBER, width=2, dash="dot")))
        arrow(def_x, 2, step_x, 2, AMBER, width=3)
        label(step_x, 65.5, "+7 m STEP-UP", AMBER)
    if wm["Our_Pressing_Intensity"] == "High":
        for y in (16, 34, 52):
            arrow(mid_x + 1, y, mid_x + 11, y, LIME, width=3)
        label(mid_x + 6, 60, "HIGH PRESS", LIME)
    if DX_PRESSING in fired:
        zx0 = min(mid_x + 6, 78.0)
        zx1 = min(zx0 + 16, 101.0)
        glow(zx0, zx1, 22, 46)
        label((zx0 + zx1) / 2, 50, "UNPRESSED BUILD-UP", RED)
        for y in (26, 34, 42):
            arrow(min(mid_x + 38, 101), y, zx0 - 1, y, RED)
    if DX_THREAT in fired:
        glow(0.5, 24, 8, 60, k=1.4)
        arrow(34, 34, 1.5, 34, RED, width=5)
        label(26, 63.5, f"⚠ IMMINENT GOAL THREAT · xG {wm['Opponent_xG_Created']:.2f}", RED, anchor="left")

    strike_x = min(mid_x + 22, 98.0)
    pts = ([(3, 34, "GK")]
           + [(def_x, y, r) for y, r in zip((10, 25, 43, 58), ("LB", "CB", "CB", "RB"))]
           + [(mid_x, y, r) for y, r in zip((12, 27, 41, 56), ("LM", "CM", "CM", "RM"))]
           + [(strike_x, y, "ST") for y in (27, 41)])
    fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", showlegend=False,
                             text=[p[2] for p in pts], hoverinfo="text",
                             marker=dict(size=15, color=SKY, line=dict(color="white", width=2))))

    if DX_PRESS_RESIST in fired:
        line_trace([def_x, def_x, def_x + 9, def_x], [25, 43, 34, 25], color=AMBER, width=2, dash="dash")
        label(def_x + 10, 34, "PASSING TRIANGLE", AMBER, anchor="left")

    anchors = {"Center Back": (def_x, 25), "Defensive Mid": (mid_x - 5, 34), "Playmaker": (mid_x + 5, 34),
               "Winger": (mid_x + 8, 60), "Striker": (strike_x, 41)}
    px, py = anchors.get(wm["Player_Position"], (mid_x, 34))
    critical = bool({DX_EXHAUSTION, DX_INJURY} & fired)
    ring = RED if critical else "#FB923C" if wm["Player_Sprint_Fatigue"] > 70 else YELLOW
    fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers", showlegend=False, hoverinfo="text",
                             text=[f"{wm['Player_Name']} · {wm['Player_Position']}<br>Fatigue {wm['Player_Sprint_Fatigue']}%"
                                   f"<br>Pass acc. under press {wm['Pass_Accuracy_Under_Press']}%"],
                             marker=dict(size=17, color=ring, line=dict(color="white", width=2))))
    fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=32, symbol="circle-open", line=dict(color=ring, width=4))))
    label(px, py - 6.5, f"{esc(wm['Player_Name'])} · {wm['Player_Sprint_Fatigue']}%", ring)

    fig.update_layout(
        shapes=shapes, annotations=anns, height=560, margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-4, 109], fixedrange=True, constrain="domain"),
        yaxis=dict(visible=False, range=[-9, 76], fixedrange=True, scaleanchor="x", scaleratio=1),
        hoverlabel=dict(bgcolor="#111827", font_color="#E5E7EB"),
    )
    return fig


PITCH_LEGEND = (
    '<div class="legend">'
    '<span><i style="background:#38BDF8"></i>Our players</span>'
    '<span><i style="background:#FDE047"></i>Analysed player (red ring = critical)</span>'
    '<span><i style="background:rgba(239,68,68,.6)"></i>Vulnerability zone / opponent threat</span>'
    '<span><i style="background:#FBBF24"></i>Suggested adjustment</span>'
    '<span><i style="background:#ECFCCB"></i>Our press</span></div>'
)


# ===========================================================================
# 4. TIMELINE CHARTS
# ===========================================================================
def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.14, x=0), font=dict(family="Inter, sans-serif", color="#E5E7EB"))
    fig.update_xaxes(gridcolor="#1F2937", zeroline=False)
    fig.update_yaxes(gridcolor="#1F2937", zeroline=False)
    return fig


def _hline(fig, y, yref, color, text):
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref=yref, y0=y, y1=y, line=dict(color=color, width=1, dash="dot"))
    fig.add_annotation(xref="paper", x=0.005, yref=yref, y=y, text=text, showarrow=False, yshift=9,
                       xanchor="left", font=dict(color=color, size=10))


def _vline(fig, x):
    fig.add_shape(type="line", xref="x", x0=x, x1=x, yref="paper", y0=0, y1=1, line=dict(color="#FDE047", width=1, dash="dash"))


XG_LINE = 0.75


def fatigue_xg_chart(df: pd.DataFrame, summary: pd.DataFrame, minute: int) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Minute"], y=df["Opponent_xG_Created"], name="Opponent xG (per 15')", width=5,
                         marker_color="rgba(239,68,68,.6)"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["Minute"], y=df["Player_Sprint_Fatigue"], name="Sprint fatigue %", mode="lines+markers",
                             line=dict(color="#10B981", width=3), marker=dict(size=8)), secondary_y=False)
    worst = summary.dropna(subset=["Worst"])
    fig.add_trace(go.Scatter(x=worst["Minute"], y=[108] * len(worst), mode="markers", name="Alerts",
                             marker=dict(symbol="diamond", size=12, color=[SEV_COLOR[s] for s in worst["Worst"]],
                                         line=dict(color="white", width=1)),
                             text=worst["Diagnoses"], hovertemplate="%{text}<extra>Alerts</extra>"), secondary_y=False)
    _hline(fig, 85, "y", "#F59E0B", "Fatigue 85% (R2)")
    _hline(fig, XG_LINE, "y2", "#F87171", f"xG {XG_LINE} (R8)")
    _vline(fig, minute)
    fig.update_yaxes(title_text="Sprint fatigue (%)", range=[0, 116], secondary_y=False)
    fig.update_yaxes(title_text="Opponent xG", range=[0, max(1.3, float(df["Opponent_xG_Created"].max()) * 1.25)],
                     showgrid=False, secondary_y=True)
    fig.update_xaxes(title_text="Match minute", tickmode="array", tickvals=df["Minute"].tolist(),
                     range=[df["Minute"].min() - 8, df["Minute"].max() + 8])
    fig.update_layout(title=dict(text="Fatigue & opponent xG progression", font=dict(size=14)))
    return style_fig(fig)


def pass_loss_chart(df: pd.DataFrame, minute: int) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["Minute"], y=df["Ball_Loss_Count"], name="Ball losses (per 15')", width=5,
                         marker_color="rgba(245,158,11,.55)"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["Minute"], y=df["Pass_Accuracy_Under_Press"], name="Pass accuracy under press %",
                             mode="lines+markers", line=dict(color="#38BDF8", width=3), marker=dict(size=8)), secondary_y=False)
    _hline(fig, 70, "y", "#F87171", "Pass accuracy 70% (R7)")
    _vline(fig, minute)
    fig.update_yaxes(title_text="Pass accuracy (%)", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text="Ball losses", range=[0, 16], showgrid=False, secondary_y=True)
    fig.update_xaxes(title_text="Match minute", tickmode="array", tickvals=df["Minute"].tolist(),
                     range=[df["Minute"].min() - 8, df["Minute"].max() + 8])
    fig.update_layout(title=dict(text="Press resistance: passing vs turnovers", font=dict(size=14)))
    return style_fig(fig)


def summarize(df: pd.DataFrame, results: list) -> pd.DataFrame:
    rows = []
    for rec, res in zip(df.to_dict("records"), results):
        worst = min((f.rule.severity for f in res.fired), key=lambda s: SEV_ORDER[s], default=None)
        rows.append({"Minute": int(rec["Minute"]), "Fatigue %": rec["Player_Sprint_Fatigue"],
                     "Pass acc. %": rec["Pass_Accuracy_Under_Press"], "Distance km": rec["Distance_Covered_km"],
                     "Ball losses": rec["Ball_Loss_Count"], "Opp. xG": rec["Opponent_xG_Created"],
                     "Alerts": len(res.fired), "Worst": worst,
                     "Diagnoses": " · ".join(f.rule.diagnosis for f in res.fired) or "—"})
    return pd.DataFrame(rows)


# ===========================================================================
# 5. SCENARIOS + SIDEBAR
# ===========================================================================
SCENARIOS: dict[str, dict | None] = {
    "Custom (manual input)": None,
    "Balanced match - no alerts": dict(
        Player_Name="A. Moreno", Player_Position="Playmaker", Player_Sprint_Fatigue=45, Pass_Accuracy_Under_Press=84,
        Distance_Covered_km=6.2, Ball_Loss_Count=2, Opponent_Build_Up_Time_s=14.0, Our_PPDA=9.0, Space_Between_Lines=11.0,
        Opponent_Through_Ball_Count=0, Opponent_xG_Created=0.05, Midfield_Line_Height_m=50.0, Substitutes_Available=3),
    "Opponent counter-attacks, weak press": dict(
        Player_Name="K. Daniels", Player_Position="Defensive Mid", Player_Sprint_Fatigue=60, Pass_Accuracy_Under_Press=75,
        Distance_Covered_km=7.0, Ball_Loss_Count=3, Opponent_Build_Up_Time_s=6.5, Our_PPDA=14.0, Space_Between_Lines=12.0,
        Opponent_Through_Ball_Count=1, Opponent_xG_Created=0.20, Midfield_Line_Height_m=44.0, Substitutes_Available=3),
    "Tired playmaker, press-resistance failure": dict(
        Player_Name="A. Moreno", Player_Position="Playmaker", Player_Sprint_Fatigue=91, Pass_Accuracy_Under_Press=58,
        Distance_Covered_km=11.9, Ball_Loss_Count=10, Opponent_Build_Up_Time_s=12.0, Our_PPDA=9.0, Space_Between_Lines=10.0,
        Opponent_Through_Ball_Count=1, Opponent_xG_Created=0.10, Midfield_Line_Height_m=45.0, Substitutes_Available=2),
    "Open central corridor + xG spike": dict(
        Player_Name="L. Haas", Player_Position="Center Back", Player_Sprint_Fatigue=70, Pass_Accuracy_Under_Press=72,
        Distance_Covered_km=9.5, Ball_Loss_Count=5, Opponent_Build_Up_Time_s=11.0, Our_PPDA=10.0, Space_Between_Lines=19.0,
        Opponent_Through_Ball_Count=5, Opponent_xG_Created=0.92, Midfield_Line_Height_m=40.0, Substitutes_Available=3),
    "Disjointed high press": dict(
        Player_Name="R. Kovac", Player_Position="Winger", Player_Sprint_Fatigue=55, Pass_Accuracy_Under_Press=80,
        Distance_Covered_km=8.0, Ball_Loss_Count=2, Opponent_Build_Up_Time_s=13.0, Our_PPDA=7.0, Space_Between_Lines=17.5,
        Opponent_Through_Ball_Count=2, Opponent_xG_Created=0.30, Midfield_Line_Height_m=56.0, Substitutes_Available=3),
    "Perfect storm (everything wrong)": dict(
        Player_Name="A. Moreno", Player_Position="Defensive Mid", Player_Sprint_Fatigue=92, Pass_Accuracy_Under_Press=58,
        Distance_Covered_km=12.1, Ball_Loss_Count=10, Opponent_Build_Up_Time_s=6.0, Our_PPDA=15.0, Space_Between_Lines=18.0,
        Opponent_Through_Ball_Count=5, Opponent_xG_Created=1.10, Midfield_Line_Height_m=30.0, Substitutes_Available=2),
}

for _k, _v in DEFAULT_FACTS.items():
    st.session_state.setdefault(_k, _v)


def load_scenario() -> None:
    preset = SCENARIOS[st.session_state["scenario"]]
    if preset:
        for k, v in preset.items():
            st.session_state[k] = cast_fact(k, v) if k in FACT_SPECS else v


def spec_slider(name: str) -> None:
    s = FACT_SPECS[name]
    st.slider(s["label"], s["min"], s["max"], step=s["step"], key=name)


MODES = ["Demo match (sample data)", "Manual simulation", "Upload CSV"]

with st.sidebar:
    st.markdown('<div class="brand"><div class="dot">⚽</div><div>PitchSense <span class="ai">AI</span></div></div>', unsafe_allow_html=True)
    mode = st.radio("Data source", MODES, key="mode")
    st.download_button("⬇ Download sample CSV", sample_csv_bytes(), "pitchsense_match_sample.csv", "text/csv")
    st.caption("CSV columns: " + ", ".join(CSV_COLUMNS))
    st.divider()

    timeline_df: pd.DataFrame | None = None
    if mode == "Manual simulation":
        st.selectbox("Demo scenario", list(SCENARIOS), key="scenario", on_change=load_scenario)
        with st.expander("👤 Player", expanded=True):
            st.text_input("Player name", key="Player_Name")
            st.selectbox("Position", POSITIONS, key="Player_Position")
            for n in ("Player_Sprint_Fatigue", "Pass_Accuracy_Under_Press", "Distance_Covered_km", "Ball_Loss_Count"):
                spec_slider(n)
        with st.expander("🎯 Opponent", expanded=True):
            for n in ("Opponent_Build_Up_Time_s", "Opponent_Through_Ball_Count", "Opponent_xG_Created"):
                spec_slider(n)
        with st.expander("🧩 Our team shape", expanded=True):
            for n in ("Our_PPDA", "Space_Between_Lines", "Midfield_Line_Height_m", "Substitutes_Available"):
                spec_slider(n)
    elif mode == "Upload CSV":
        upload = st.file_uploader("Upload match tracking CSV", type="csv")
        if upload is not None:
            try:
                timeline_df, errs, warns = validate_dataframe(pd.read_csv(upload))
            except Exception as exc:
                timeline_df, errs, warns = None, [f"Could not read the file: {exc}"], []
            for e in errs:
                st.error(e)
            for w in warns:
                st.warning(w)
    else:
        timeline_df = sample_dataframe()

    if timeline_df is not None:
        players = sorted(timeline_df["Player_Name"].unique())
        player = st.selectbox("Player", players) if len(players) > 1 else players[0]
        timeline_df = timeline_df[timeline_df["Player_Name"] == player].reset_index(drop=True)
        minutes = timeline_df["Minute"].astype(int).tolist()
        minute = st.select_slider("Match minute (analysed window)", options=minutes, value=minutes[-1])

# ===========================================================================
# 6. HERO + RUN THE EXPERT SYSTEM
# ===========================================================================
MODE_PILL = {MODES[0]: "● DEMO MATCH", MODES[1]: "● MANUAL SIMULATION", MODES[2]: "● CSV ANALYSIS"}
st.markdown(
    '<div class="hero"><div><div class="eyebrow">PITCHSENSE AI · TACTICAL INTELLIGENCE SUITE</div>'
    '<div class="title">PitchSense AI — Tactical Expert System</div>'
    f'<div class="sub">Forward-chaining inference · {len(RULES)} production rules · fully explainable alerts</div></div>'
    f'<div class="pill">{MODE_PILL[mode]}</div></div>', unsafe_allow_html=True)

engine = InferenceEngine(RULES)
results: list = []
summary: pd.DataFrame | None = None

if mode == "Upload CSV" and timeline_df is None:
    st.markdown('<div class="empty">Upload a CSV in the sidebar to analyse a full match.<br>'
                'Download the sample CSV to see the expected columns.</div>', unsafe_allow_html=True)
    st.stop()

if timeline_df is not None:
    results = analyze_timeline(timeline_df, engine)
    summary = summarize(timeline_df, results)
    idx = int(timeline_df.index[timeline_df["Minute"].astype(int) == minute][0])
    result = results[idx]
else:
    result = engine.run({k: cast_fact(k, st.session_state[k]) for k in FACT_COLUMNS})

wm = result.working_memory
fired_sorted = sorted(result.fired, key=lambda f: SEV_ORDER[f.rule.severity])
fired_names = {f.rule.diagnosis for f in result.fired}

ctx = [chip("Player", wm["Player_Name"]), chip("Position", wm["Player_Position"])]
if timeline_df is not None:
    ctx.append(chip("Window", f"{minute - 15}'–{minute}'"))
ctx += [chip("Opp. build-up", f'{wm["Opponent_Build_Up_Speed"]} ({wm["Opponent_Build_Up_Time_s"]:.1f}s)',
             "bad" if wm["Opponent_Build_Up_Speed"] == "Fast" else ""),
        chip("Ball losses", wm["Ball_Loss_Turnovers"], "bad" if wm["Ball_Loss_Turnovers"] == "High" else ""),
        chip("Through balls", wm["Opponent_Through_Balls"], "bad" if wm["Opponent_Through_Balls"] == "Frequent" else ""),
        chip("Midfield line", f'{wm["Midfield_Line_Height"]} ({wm["Midfield_Line_Height_m"]:.0f} m)'),
        chip("Subs left", wm["Substitutes_Available"])]
st.markdown(f'<div class="chips">{"".join(ctx)}</div>', unsafe_allow_html=True)

worst_sev = fired_sorted[0].rule.severity if fired_sorted else None
press = wm["Our_Pressing_Intensity"]
tiles = [
    kpi("Sprint fatigue", f'{wm["Player_Sprint_Fatigue"]}%', "R2 > 85 · R9 > 80", tone_hi(wm["Player_Sprint_Fatigue"], 70, 85)),
    kpi("Pass acc. under press", f'{wm["Pass_Accuracy_Under_Press"]}%', "R7 < 70", tone_lo(wm["Pass_Accuracy_Under_Press"], 78, 70)),
    kpi("Distance covered", f'{wm["Distance_Covered_km"]:.1f} km', "R9 > 11.5 km", tone_hi(wm["Distance_Covered_km"], 10, 11.5)),
    kpi("Opponent xG (15')", f'{wm["Opponent_xG_Created"]:.2f}', "R8 > 0.75", tone_hi(wm["Opponent_xG_Created"], 0.4, 0.75)),
    kpi("Pressing intensity", press, f'PPDA {wm["Our_PPDA"]:.1f}', {"High": "ok", "Medium": "warn", "Low": "bad"}[press]),
    kpi("Active alerts", str(len(fired_sorted)), f"Highest: {worst_sev}" if worst_sev else "All clear",
        "bad" if worst_sev == "Critical" else "warn" if worst_sev in ("High", "Medium") else "ok"),
]
for col, tile in zip(st.columns(6), tiles):
    col.markdown(tile, unsafe_allow_html=True)

left, right = st.columns([1.7, 1])
with left:
    st.markdown('<div class="section">Tactical pitch · live shape & vulnerabilities</div>', unsafe_allow_html=True)
    st.plotly_chart(build_pitch(wm, fired_names), theme=None, config={"displayModeBar": False}, key="pitch")
    st.markdown(PITCH_LEGEND, unsafe_allow_html=True)
with right:
    st.markdown('<div class="section">Expert system alerts</div>', unsafe_allow_html=True)
    cards = "".join(alert_card(f) for f in fired_sorted) or \
        '<div class="empty">✅ No tactical alerts.<br>Current metrics trigger none of the rules.</div>'
    st.markdown(f'<div class="feed">{cards}</div>', unsafe_allow_html=True)

# ===========================================================================
# 7. TABS
# ===========================================================================
tab_time, tab_why, tab_trace, tab_wm, tab_kb = st.tabs(
    ["📈 Match Timeline", "💡 Explanation (Why?)", "🔁 Inference Trace", "🧠 Working Memory", "📚 Knowledge Base"])

with tab_time:
    if timeline_df is None:
        st.info("Switch the data source to **Demo match** or **Upload CSV** to see the 90-minute timeline.")
    else:
        c1, c2 = st.columns(2)
        c1.plotly_chart(fatigue_xg_chart(timeline_df, summary, minute), theme=None,
                        config={"displayModeBar": False}, key="chart_fatigue")
        c2.plotly_chart(pass_loss_chart(timeline_df, minute), theme=None,
                        config={"displayModeBar": False}, key="chart_pass")
        st.markdown('<div class="section">Alerts by interval (diamonds above = worst severity)</div>', unsafe_allow_html=True)
        st.dataframe(summary.drop(columns=["Worst"]), hide_index=True)

with tab_why:
    if not result.fired:
        st.info("No rule fired, so there is nothing to explain yet. See below why each rule did not fire.")
    for f in fired_sorted:
        with st.expander(f"{f.rule.rule_id} · {f.rule.diagnosis}  ({f.rule.severity})", expanded=True):
            st.markdown(f"**Rule:** {f.rule.name}  ·  **Fired in cycle:** {f.cycle}  ·  **Type:** {f.rule.source}")
            st.code(f.rule.as_text(), language=None, wrap_lines=True)
            st.markdown("**Evidence (why the IF-part was true):**")
            st.dataframe(pd.DataFrame([
                {"Condition": c.describe(), "Actual value": "—" if c.fact.startswith(DX_PREFIX) else str(a), "Satisfied": "✅"}
                for c, a in f.evidence]), hide_index=True)
            st.markdown(f"**Football reasoning:** {f.rule.rationale}")
            if any(c.fact.startswith(DX_PREFIX) for c, _ in f.evidence):
                st.caption("🔗 Reached by chaining on conclusions made by earlier rules.")
    st.markdown('<div class="section">Why did the other rules NOT fire?</div>', unsafe_allow_html=True)
    for rule, reasons in result.not_fired:
        with st.expander(f"{rule.rule_id} · {rule.diagnosis} (not triggered)"):
            for reason in reasons:
                st.markdown(f"- ❌ {reason}")

with tab_trace:
    st.markdown("Step-by-step forward-chaining log (abstraction → match → resolve → fire).")
    st.code("\n".join(result.trace), language=None)

with tab_wm:
    st.markdown("All facts currently in working memory and where each came from.")
    st.dataframe(pd.DataFrame([{"Fact": k, "Value": str(v), "Origin": result.origin[k]} for k, v in wm.items()]),
                 hide_index=True)

with tab_kb:
    st.markdown(f"{len(RULES)} production rules. R1-R3 are the core rules; R4-R10 extend them, several by chaining.")
    st.dataframe(pd.DataFrame([{"ID": r.rule_id, "Type": r.source, "Priority": r.priority, "Severity": r.severity,
                                "Rule": r.as_text()} for r in RULES]), hide_index=True)