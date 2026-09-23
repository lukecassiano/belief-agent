"""
Belief Agent — interactive demo (Branch A: belief inference under uncertainty).

Two agents infer a hidden goal (A / B / C) from private, noisy evidence and
exchange confidence-weighted messages. The demo makes the internals visible:
belief distributions, entropy, precision, the loss landscape they move across,
and how much of each move came from evidence versus from being told.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from instrument import (
    CORNERS, loss_surface, simplex_grid, to_xy, trace_episode, update_field,
)
from simulation import GOAL_NAMES, LOG3, run_episode_history, run_many_mode

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens — matched to portfolio.lukecassiano.com
# ─────────────────────────────────────────────────────────────────────────────
BG = "#0d0d12"
PANEL = "#14141b"
GRID = "rgba(245,243,238,0.08)"
LINE = "rgba(245,243,238,0.14)"
INK = "#f5f3ee"
MUTED = "rgba(246,244,239,0.55)"
SENSOR = "#4ade80"      # portfolio green
LANGUAGE = "#a5b4fc"    # indigo, lifted from the portfolio's #312e81 for contrast
TRUTH = "#f5c89a"       # warm cream accent
BAD = "#f87171"
MONO = "'IBM Plex Mono', ui-monospace, monospace"
DISPLAY = "'Syne', system-ui, sans-serif"

MODES = {
    "none": "No communication",
    "bidirectional": "Bidirectional  S ⇄ L",
    "unidirectional_S_to_L": "Unidirectional  S → L",
}
MODE_SHORT = {"none": "no comm", "bidirectional": "S ⇄ L", "unidirectional_S_to_L": "S → L"}
MODE_COLORS = {"none": "rgba(246,244,239,0.55)", "bidirectional": BAD, "unidirectional_S_to_L": SENSOR}
AGENT = {"S": ("Sensor", SENSOR), "L": ("Language", LANGUAGE)}

# Curated presets. Seed 36 at noise 0.4 is a clean illustration of the core
# finding: identical evidence, three wiring choices, three different outcomes.
SCENARIOS = {
    "Herding — confident noise spreads": dict(
        seed=36, noise=0.40, steps=8, mode="bidirectional",
        blurb="Symmetric trust lets the noisy language agent drag the reliable sensor agent "
              "to the wrong goal. They end up agreeing — and both are wrong.",
    ),
    "Recovery — trust the reliable source": dict(
        seed=36, noise=0.40, steps=8, mode="unidirectional_S_to_L",
        blurb="Same evidence stream, but messages only flow Sensor → Language. "
              "The weaker agent is pulled toward the truth instead of the reverse.",
    ),
    "Isolation — no communication": dict(
        seed=36, noise=0.40, steps=8, mode="none",
        blurb="Each agent reasons alone. The sensor gets it right; the language agent, "
              "working from ambiguous constraints, does not.",
    ),
    "Clean evidence — easy mode": dict(
        seed=0, noise=0.10, steps=6, mode="bidirectional",
        blurb="With low noise both channels are informative, and communication speeds up "
              "convergence at no cost.",
    ),
}

st.set_page_config(page_title="Belief Agent — Luke Cassiano", page_icon="◬", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
.stApp {{
  background:
    radial-gradient(ellipse 55% 45% at 92% -5%, rgba(74,222,128,0.16), transparent 70%),
    radial-gradient(ellipse 60% 55% at 108% 12%, rgba(49,46,129,0.55), transparent 70%),
    {BG};
  background-attachment: fixed;
}}
html, body, .stApp, .stMarkdown, .stApp p, .stApp li, .stApp label, .stCaption,
.stApp input, .stApp [data-baseweb="select"] div, .stApp [data-testid="stWidgetLabel"] {{
  font-family: {MONO} !important;
}}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: 'Material Symbols Rounded' !important; }}
.block-container {{ padding-top: 2.6rem; max-width: 1240px; }}
h1, h2, h3, h4, h5, .display {{ font-family: {DISPLAY} !important; font-weight: 800 !important;
  letter-spacing: -0.03em; color: {INK}; }}
h5 {{ font-size: 1.15rem !important; letter-spacing: -0.02em; margin-bottom: 0 !important; }}
.eyebrow {{ font-size: 0.72rem; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase;
  color: {MUTED}; margin-bottom: 0.3rem; }}
.hero-title {{ font-family: {DISPLAY}; font-weight: 800; font-size: 3.4rem; line-height: 1.02;
  letter-spacing: -0.035em; color: {INK}; margin: 0 0 0.6rem; }}
.hero-sub {{ color: {INK}; opacity: 0.85; font-size: 0.95rem; line-height: 1.65; max-width: 640px; }}
.links a {{ color: {INK} !important; text-decoration: none; font-size: 0.75rem; letter-spacing: 0.08em;
  text-transform: uppercase; margin-right: 1.4rem; }}
.links a:hover {{ color: {SENSOR} !important; }}
.chip {{ display: inline-block; font-size: 0.7rem; padding: 3px 10px; border: 1px solid {LINE};
  border-radius: 999px; color: {MUTED}; margin: 0 6px 6px 0; }}
.card {{ background: rgba(20,20,27,0.72); border: 1px solid {LINE}; border-radius: 12px;
  padding: 14px 16px; backdrop-filter: blur(6px); height: 100%; }}
.card .label {{ font-size: 0.66rem; letter-spacing: 0.16em; text-transform: uppercase; color: {MUTED}; }}
.card .value {{ font-family: {DISPLAY}; font-weight: 800; font-size: 1.7rem; line-height: 1.2; }}
.card .note {{ font-size: 0.74rem; color: {MUTED}; }}
.blurb {{ border-left: 2px solid {SENSOR}; padding: 8px 14px; color: {INK}; font-size: 0.88rem;
  background: rgba(74,222,128,0.05); border-radius: 0 8px 8px 0; margin: 4px 0 18px; }}
.msg {{ font-size: 0.8rem; }}
[data-testid="stSidebar"] {{ background: rgba(13,13,18,0.92); border-right: 1px solid {LINE}; }}
[data-baseweb="tab-list"] {{ gap: 1.4rem; }}
[data-baseweb="tab"] p {{ font-size: 0.74rem !important; letter-spacing: 0.12em; text-transform: uppercase; }}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation wrappers (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def episode(seed, steps, noise, mode, flat_precision):
    hist, summary = run_episode_history(np.random.default_rng(int(seed)), steps=int(steps),
                                        noise=float(noise), mode=mode, flat_precision=flat_precision)
    _, trace = trace_episode(seed, steps, noise, mode, flat_precision)
    for h in hist:
        h["S_belief"] = [float(x) for x in h["S_belief"]]
        h["L_belief"] = [float(x) for x in h["L_belief"]]
    return hist, summary, trace


@st.cache_data(show_spinner=False)
def sweep(noise_values, steps, episodes, n_seeds, flat_precision):
    rows = []
    for nz in noise_values:
        for mode in MODES:
            per_seed = [run_many_mode(seed=1000 + s, episodes=int(episodes), steps=int(steps),
                                      noise=float(nz), mode=mode, flat_precision=flat_precision)
                        for s in range(int(n_seeds))]
            for metric in ("S_correct", "L_correct", "both_correct", "agree"):
                vals = [d[metric] for d in per_seed]
                rows.append(dict(noise=nz, mode=mode, metric=metric,
                                 mean=float(np.mean(vals)), std=float(np.std(vals))))
    return pd.DataFrame(rows)


def with_origin(hist):
    u = [1 / 3] * 3
    return [dict(t=0, S_belief=u, L_belief=u, S_entropy=LOG3, L_entropy=LOG3)] + hist


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def style(fig, height=420):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        font=dict(family="IBM Plex Mono, monospace", color=INK, size=11),
        margin=dict(l=36, r=24, t=40, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(family="IBM Plex Mono", color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_annotations(selector=dict(showarrow=False), font_color=MUTED)
    return fig


LOSS_SCALE = [[0.0, "rgba(74,222,128,0.55)"], [0.18, "rgba(74,222,128,0.18)"],
              [0.5, "rgba(49,46,129,0.55)"], [1.0, "rgba(13,13,18,0.0)"]]


def simplex_canvas(true_goal, show_loss=True):
    """Triangle with the loss landscape −ln b(true) as a contour field."""
    fig = go.Figure()
    if show_loss:
        xs, ys, Z = loss_surface(true_goal)
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=Z, zmin=0, zmax=4, ncontours=28, colorscale=LOSS_SCALE,
            contours=dict(coloring="heatmap", showlines=True), line=dict(color="rgba(245,243,238,0.05)", width=0.5),
            showscale=False, hoverinfo="skip", connectgaps=False))
    tri = np.vstack([CORNERS, CORNERS[:1]])
    fig.add_trace(go.Scatter(x=tri[:, 0], y=tri[:, 1], mode="lines", line=dict(color=LINE, width=1.2),
                             hoverinfo="skip", showlegend=False))
    offs = [(0, 0.06), (-0.05, -0.05), (0.05, -0.05)]
    for g in range(3):
        is_true = g == true_goal
        fig.add_annotation(x=CORNERS[g, 0] + offs[g][0], y=CORNERS[g, 1] + offs[g][1],
                           text=f"<b>{GOAL_NAMES[g]}</b>" + (" ★" if is_true else ""), showarrow=False,
                           font=dict(family="Syne, sans-serif", size=16, color=TRUTH if is_true else MUTED))
    fig.update_xaxes(visible=False, range=[-0.1, 1.1])
    fig.update_yaxes(visible=False, range=[-0.1, 0.96], scaleanchor="x", scaleratio=1)
    return fig


# 2-D goal environment: each goal is a point on a grid, and an agent's position is
# its belief-weighted average of the goal positions (same mapping as the original demo).
GOAL_POS = np.array([[0.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])   # A, B, C


def belief_to_grid(b):
    return np.asarray(b, dtype=float) @ GOAL_POS


def grid_canvas(true_goal):
    """Plain grid with the three goals as nodes — no loss shading."""
    fig = go.Figure()
    for g in range(3):
        is_true = g == true_goal
        x, y = GOAL_POS[g]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(size=34, color="rgba(245,200,154,0.14)" if is_true else "rgba(245,243,238,0.05)",
                        line=dict(color=TRUTH if is_true else LINE, width=1.5))))
        fig.add_annotation(x=x, y=y + (0.3 if g == 0 else -0.3), showarrow=False,
                           text=f"<b>{GOAL_NAMES[g]}</b>" + (" ★" if is_true else ""),
                           font=dict(family="Syne, sans-serif", size=16, color=TRUTH if is_true else MUTED))
    fig.update_xaxes(range=[-1.45, 1.45], dtick=0.25, showgrid=True, gridcolor=GRID, zeroline=False,
                     showticklabels=False, showline=True, linecolor=LINE, mirror=True)
    fig.update_yaxes(range=[-1.45, 1.45], dtick=0.25, showgrid=True, gridcolor=GRID, zeroline=False,
                     showticklabels=False, showline=True, linecolor=LINE, mirror=True,
                     scaleanchor="x", scaleratio=1)
    return fig


def trajectory_traces(S_xy, L_xy, k):
    out = []
    for xy, (name, color), sym in ((S_xy, AGENT["S"], "circle"), (L_xy, AGENT["L"], "square")):
        out.append(go.Scatter(x=xy[: k + 1, 0], y=xy[: k + 1, 1], mode="lines+markers",
                              line=dict(color=color, width=2.2), marker=dict(size=4, color=color),
                              opacity=0.75, hoverinfo="skip", showlegend=False))
        out.append(go.Scatter(x=[xy[k, 0]], y=[xy[k, 1]], mode="markers", name=name,
                              marker=dict(color=color, size=15, symbol=sym, line=dict(color=BG, width=2)),
                              hovertemplate=f"{name} · t={k}<extra></extra>"))
    return out


def grid_figure(hist, true_goal, height=500, animate=True, title=None):
    """Agents move on a grid toward whichever goal they believe in; trails show the path."""
    steps = with_origin(hist)
    S_xy = np.array([belief_to_grid(s["S_belief"]) for s in steps])
    L_xy = np.array([belief_to_grid(s["L_belief"]) for s in steps])
    n = len(steps)
    fig = grid_canvas(true_goal)
    base = len(fig.data)
    for tr in trajectory_traces(S_xy, L_xy, n - 1):
        fig.add_trace(tr)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(family="Syne, sans-serif", size=15, color=INK),
                                     x=0.03), showlegend=False)
    if animate:
        idx = list(range(base, base + 4))
        fig.frames = [go.Frame(data=trajectory_traces(S_xy, L_xy, k), traces=idx, name=str(k))
                      for k in range(n)]
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", direction="left", x=0.0, y=0.0, xanchor="left", yanchor="top",
                bgcolor=PANEL, bordercolor=LINE, font=dict(color=INK, size=11), showactive=False,
                buttons=[dict(label="▶ Replay", method="animate",
                              args=[[str(k) for k in range(n)],
                                    dict(frame=dict(duration=600, redraw=False),
                                         transition=dict(duration=300, easing="cubic-in-out"),
                                         mode="immediate")])])],
            sliders=[dict(
                active=n - 1, x=0.2, y=0.03, len=0.8, xanchor="left", yanchor="top", pad=dict(t=0),
                bgcolor=PANEL, activebgcolor=SENSOR, bordercolor=LINE, tickcolor=MUTED,
                font=dict(color=MUTED, size=10),
                currentvalue=dict(prefix="t = ", font=dict(color=INK, size=12)),
                steps=[dict(label=str(k), method="animate",
                            args=[[str(k)], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                 transition=dict(duration=0))]) for k in range(n)])])
    style(fig, height=height)
    fig.update_layout(margin=dict(l=10, r=10, t=50 if title else 30, b=80 if animate else 10))
    return fig


def dynamics_figure(hist, true_goal, height=500):
    steps = with_origin(hist)
    t = [s["t"] for s in steps]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=("P(hidden goal)", "Entropy  ·  nats, max = ln 3"))
    for key, (name, color) in AGENT.items():
        fig.add_trace(go.Scatter(x=t, y=[s[f"{key}_belief"][true_goal] for s in steps], name=name,
                                 mode="lines+markers", line=dict(color=color, width=2.5, shape="spline"),
                                 marker=dict(size=6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=[s[f"{key}_entropy"] for s in steps], name=name, showlegend=False,
                                 mode="lines+markers", line=dict(color=color, width=2.5, shape="spline"),
                                 marker=dict(size=6)), row=2, col=1)
    fig.add_hline(y=1 / 3, line=dict(color=MUTED, dash="dot", width=1), row=1, col=1,
                  annotation_text="chance", annotation_font_color=MUTED, annotation_font_size=10)
    fig.update_yaxes(range=[0, 1.02], row=1, col=1)
    fig.update_yaxes(range=[0, LOG3 * 1.05], row=2, col=1)
    fig.update_xaxes(title_text="timestep", row=2, col=1, dtick=1)
    style(fig, height=height)
    fig.update_layout(legend=dict(y=1.1), margin=dict(t=70))
    return fig


def field_figure(trace, true_goal, t, receiver, source, weight_scale, height=540):
    """Vector field of the update rule for one step, drawn over the loss landscape."""
    r = trace[t - 1]
    if source == "message":
        like = r[f"{receiver}_msg_like_in"]
        w = r[f"{receiver}_w_in"] * weight_scale
        start, end = r[f"{receiver}_priv"], r[f"{receiver}_post"]
    else:
        like = r[f"{receiver}_like"]
        w = 1.0 * weight_scale
        start, end = r[f"{receiver}_prior"], r[f"{receiver}_priv"]

    fig = simplex_canvas(true_goal)
    name, color = AGENT[receiver]
    if like is None or w <= 1e-9:
        fig.add_annotation(x=0.5, y=0.3, text="no incoming message in this wiring", showarrow=False,
                           font=dict(color=MUTED, size=13))
    else:
        P = simplex_grid(20)
        D = update_field(P, like, w)
        XY0, XY1 = P @ CORNERS, (P + D) @ CORNERS
        mag = np.linalg.norm(XY1 - XY0, axis=1)
        scale = 0.9 * (1 / 20) / max(mag.max(), 1e-9)   # longest arrow ≈ one grid cell
        tip = XY0 + (XY1 - XY0) * scale
        xs, ys = [], []
        for (x0, y0), (x1, y1) in zip(XY0, tip):
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(245,243,238,0.45)", width=1.1),
                                 hoverinfo="skip", showlegend=False))
        ang = np.degrees(np.arctan2(tip[:, 0] - XY0[:, 0], tip[:, 1] - XY0[:, 1]))
        rel = mag / max(mag.max(), 1e-9)
        fig.add_trace(go.Scatter(x=tip[:, 0], y=tip[:, 1], mode="markers", showlegend=False,
                                 marker=dict(symbol="triangle-up", angle=ang, size=4 + 5 * rel, color=rel,
                                             colorscale=[[0, "rgba(245,243,238,0.35)"], [1, TRUTH]]),
                                 hoverinfo="skip"))
    a, b = to_xy(start), to_xy(end)
    if np.linalg.norm(b - a) > 1e-4:
        fig.add_annotation(x=b[0], y=b[1], ax=a[0], ay=a[1], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=3, arrowcolor=color, text="")
    fig.add_trace(go.Scatter(x=[a[0]], y=[a[1]], mode="markers", name=f"{name} before",
                             marker=dict(size=10, color=BG, line=dict(color=color, width=2))))
    fig.add_trace(go.Scatter(x=[b[0]], y=[b[1]], mode="markers", name=f"{name} after",
                             marker=dict(size=13, color=color, line=dict(color=BG, width=2))))
    style(fig, height=height)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig, like, w


def descent_figure(trace, height=470):
    """Loss over time, and its per-step slope split into evidence vs. message."""
    t = [0] + [r["t"] for r in trace]
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.16, horizontal_spacing=0.1,
                        subplot_titles=("Sensor loss", "Language loss", "Sensor ΔL / step", "Language ΔL / step"))
    for col, key in ((1, "S"), (2, "L")):
        name, color = AGENT[key]
        Ls = [float(LOG3)] + [r[f"{key}_loss_post"] for r in trace]
        fill = "rgba(74,222,128,0.07)" if key == "S" else "rgba(165,180,252,0.08)"
        fig.add_trace(go.Scatter(x=t, y=Ls, mode="lines+markers", line=dict(color=color, width=2.5, shape="spline"),
                                 fill="tozeroy", fillcolor=fill, name=name, showlegend=False,
                                 hovertemplate="t=%{x}<br>L=%{y:.2f}<extra></extra>"), row=1, col=col)
        ev = [r[f"{key}_dL_evidence"] for r in trace]
        ms = [r[f"{key}_dL_message"] for r in trace]
        fig.add_trace(go.Bar(x=t[1:], y=ev, name="own evidence", legendgroup="ev", showlegend=(col == 1),
                             marker_color="rgba(245,243,238,0.35)",
                             hovertemplate="evidence ΔL %{y:+.2f}<extra></extra>"), row=2, col=col)
        fig.add_trace(go.Bar(x=t[1:], y=ms, name="messages (red = uphill)", legendgroup="ms",
                             showlegend=(col == 1), marker_color=[BAD if v > 1e-6 else SENSOR for v in ms],
                             hovertemplate="message ΔL %{y:+.2f}<extra></extra>"), row=2, col=col)
    fig.update_layout(barmode="relative")
    fig.update_xaxes(dtick=1)
    style(fig, height=height)
    fig.update_layout(legend=dict(y=1.12), margin=dict(t=80))
    return fig


def sweep_figure(df):
    titles = {"S_correct": "Sensor correct", "L_correct": "Language correct",
              "both_correct": "Both correct", "agree": "Agents agree"}
    fig = make_subplots(rows=1, cols=4, subplot_titles=list(titles.values()), horizontal_spacing=0.05)
    fills = {"none": "rgba(246,244,239,0.10)", "bidirectional": "rgba(248,113,113,0.15)",
             "unidirectional_S_to_L": "rgba(74,222,128,0.15)"}
    for i, metric in enumerate(titles, start=1):
        for mode in MODES:
            d = df[(df.metric == metric) & (df["mode"] == mode)].sort_values("noise")
            fig.add_trace(go.Scatter(
                x=list(d.noise) + list(d.noise[::-1]),
                y=list(d["mean"] + d["std"]) + list((d["mean"] - d["std"])[::-1]),
                mode="lines", fill="toself", fillcolor=fills[mode], line=dict(width=0),
                hoverinfo="skip", showlegend=False), row=1, col=i)
            fig.add_trace(go.Scatter(
                x=d.noise, y=d["mean"], name=MODES[mode], legendgroup=mode, showlegend=(i == 1),
                mode="lines+markers", line=dict(color=MODE_COLORS[mode], width=2.5), marker=dict(size=5),
                hovertemplate="noise %{x:.1f}<br>%{y:.1%}<extra>" + MODE_SHORT[mode] + "</extra>"),
                row=1, col=i)
        fig.update_yaxes(range=[0, 1.02], tickformat=".0%", row=1, col=i, showticklabels=(i == 1))
        fig.update_xaxes(title_text="evidence noise", row=1, col=i)
    style(fig, height=400)
    fig.update_layout(legend=dict(y=1.16), margin=dict(t=90))
    return fig


def card(label, value, note="", color=INK):
    st.markdown(f"<div class='card'><div class='label'>{label}</div>"
                f"<div class='value' style='color:{color}'>{value}</div>"
                f"<div class='note'>{note}</div></div>", unsafe_allow_html=True)


PLOT_CFG = {"displayModeBar": False}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='eyebrow'>Scenario</div>", unsafe_allow_html=True)
    scenario = st.selectbox("Scenario", list(SCENARIOS) + ["Custom"], label_visibility="collapsed")
    preset = SCENARIOS.get(scenario)
    if preset:
        st.caption(preset["blurb"])
    d = preset or dict(seed=42, noise=0.30, steps=6, mode="bidirectional")

    st.markdown("<div class='eyebrow' style='margin-top:1.2rem'>Environment</div>", unsafe_allow_html=True)
    noise = st.slider("Evidence noise", 0.0, 0.6, float(d["noise"]), 0.05,
                      help="Probability a private observation points to the wrong goal.")
    steps = st.slider("Timesteps", 1, 12, int(d["steps"]), 1)
    seed = st.number_input("Random seed", value=int(d["seed"]), step=1,
                           help="Same seed → same hidden goal and same evidence stream.")

    st.markdown("<div class='eyebrow' style='margin-top:1.2rem'>Communication</div>", unsafe_allow_html=True)
    mode = st.radio("Wiring", list(MODES), index=list(MODES).index(d["mode"]), format_func=MODES.get,
                    label_visibility="collapsed")
    with st.expander("Precision weighting"):
        precision_mode = st.radio("Trust in incoming messages", ["dynamic  1 − H/ln 3", "flat"], index=0)
        flat_p = st.slider("Flat precision", 0.0, 1.0, 0.5, 0.05, disabled=precision_mode != "flat")
    flat_precision = None if precision_mode != "flat" else float(flat_p)

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class='eyebrow'>02 — Research · Live demo</div>
<div class='hero-title'>Belief Agent</div>
<div class='hero-sub'>Two agents infer a hidden goal from private, noisy evidence and trade
confidence-weighted messages. Every internal state is exposed — belief, entropy, precision, loss —
so you can watch herding, collapse and recovery happen, and see which moves came from evidence
and which came from being told.</div>
<div class='links' style='margin:1rem 0 0.8rem'>
  <a href='https://portfolio.lukecassiano.com/belief-agent' target='_blank'>Case study →</a>
  <a href='https://github.com/lukecassiano/belief-agent' target='_blank'>GitHub →</a>
</div>
<span class='chip'><span style='color:{SENSOR}'>●</span> Sensor · reliable beacon</span>
<span class='chip'><span style='color:{LANGUAGE}'>■</span> Language · ambiguous constraints</span>
<span class='chip'><span style='color:{TRUTH}'>★</span> hidden goal</span>
""",
    unsafe_allow_html=True,
)
st.write("")

hist, summary, trace = episode(seed, steps, noise, mode, flat_precision)
true_goal = summary["true"]
preset_active = bool(preset) and (int(seed), round(noise, 2), int(steps), mode) == (
    preset["seed"], round(preset["noise"], 2), preset["steps"], preset["mode"])

tab_ep, tab_grad, tab_cmp, tab_sweep, tab_model = st.tabs(
    ["Episode", "Gradient field & cascades", "Same evidence, three wirings", "Noise sweep", "How it works"])

# ─────────────────────────────────────────────────────────────────────────────
# Episode
# ─────────────────────────────────────────────────────────────────────────────
with tab_ep:
    if preset_active:
        st.markdown(f"<div class='blurb'>{preset['blurb']}</div>", unsafe_allow_html=True)
    c = st.columns(4)
    with c[0]:
        card("Hidden goal", GOAL_NAMES[true_goal], f"seed {seed} · noise {noise:.2f}", TRUTH)
    with c[1]:
        ok = summary["S_correct"]
        card("Sensor concludes", GOAL_NAMES[summary["pred_S"]], "correct" if ok else "wrong", SENSOR if ok else BAD)
    with c[2]:
        ok = summary["L_correct"]
        card("Language concludes", GOAL_NAMES[summary["pred_L"]], "correct" if ok else "wrong",
             LANGUAGE if ok else BAD)
    with c[3]:
        if summary["agree"] and not summary["both_correct"]:
            card("Consensus", "Herded", "agree, but wrong", BAD)
        elif summary["both_correct"]:
            card("Consensus", "Aligned", "agree, and right", SENSOR)
        else:
            card("Consensus", "Split", "agents disagree", MUTED)

    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("##### Goal environment")
        st.caption("Each agent sits at its belief-weighted average of the goal positions, so it drifts "
                   "toward the goal it believes in and reaches a node only when certain. Scrub or press ▶ Replay.")
        st.plotly_chart(grid_figure(hist, true_goal), width="stretch", config=PLOT_CFG)
    with right:
        st.markdown("##### Confidence & doubt")
        st.caption("Top: probability each agent puts on the hidden goal. Bottom: Shannon entropy.")
        st.plotly_chart(dynamics_figure(hist, true_goal), width="stretch", config=PLOT_CFG)

    st.markdown("##### Step log")
    st.caption("What each agent saw privately, what it said, and how sure it was.")
    rows = [{
        "t": s["t"],
        "sensor sees": f"points to {GOAL_NAMES[s['obs']]}" + ("" if s["obs"] == true_goal else "  ✗"),
        "language hears": s["clue"],
        "S says": s["S_msg"], "L says": s["L_msg"],
        "S precision": s["S_precision"], "L precision": s["L_precision"],
        "S · P(A,B,C)": " ".join(f"{p:.2f}" for p in s["S_belief"]),
        "L · P(A,B,C)": " ".join(f"{p:.2f}" for p in s["L_belief"]),
    } for s in hist]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "S precision": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        "L precision": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
    })

# ─────────────────────────────────────────────────────────────────────────────
# Gradient field & cascades
# ─────────────────────────────────────────────────────────────────────────────
with tab_grad:
    st.markdown(
        "<div class='blurb'>Every update is a move across the probability simplex. The arrows show where "
        "the update rule <i>would</i> push any belief at this step; the shading is the loss −ln P(true). "
        "A healthy update runs downhill. An <b>information cascade</b> is a message that pushes an agent "
        "<b>uphill</b>, against the truth, with enough weight to override its own evidence.</div>",
        unsafe_allow_html=True)

    # default to the step where messages did the most damage, so the story is visible on load
    # default to the first real cascade (a message that climbs the loss), else the worst step
    firsts = [r for r in trace if max(r["S_dL_message"], r["L_dL_message"]) > 0.25]
    worst = firsts[0] if firsts else max(trace, key=lambda r: max(r["S_dL_message"], r["L_dL_message"]))
    worst_agent = "S" if worst["S_dL_message"] >= worst["L_dL_message"] else "L"
    ctl = st.columns([1.2, 1, 1.1, 1.3])
    t_sel = ctl[0].slider("Timestep", 1, len(trace), int(worst["t"]), key=f"t_{seed}_{steps}_{noise}_{mode}")
    receiver = ctl[1].radio("Agent", ["S", "L"], index=0 if worst_agent == "S" else 1,
                            format_func=lambda k: AGENT[k][0], horizontal=True,
                            key=f"r_{seed}_{steps}_{noise}_{mode}")
    source = ctl[2].radio("Force", ["message", "evidence"], horizontal=True,
                          format_func=lambda s: "incoming message" if s == "message" else "own evidence")
    w_scale = ctl[3].slider("Decision-weight multiplier", 0.0, 3.0, 1.0, 0.1,
                            help="Scales the fusion exponent w. 1.0 = what the agent actually used. "
                                 "Drag it to see how trust stretches or flattens the field.")

    g_left, g_right = st.columns([1.05, 1])
    with g_left:
        fig, like, w = field_figure(trace, true_goal, t_sel, receiver, source, w_scale)
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    with g_right:
        r = trace[t_sel - 1]
        name, color = AGENT[receiver]
        if source == "message":
            before, after = r[f"{receiver}_loss_priv"], r[f"{receiver}_loss_post"]
            sender = "L" if receiver == "S" else "S"
            said = r[f"{sender}_msg"] if like is not None else "—"
            what = f"hears {AGENT[sender][0]} say <b>“{said}”</b>"
        else:
            before, after = r[f"{receiver}_loss_prior"], r[f"{receiver}_loss_priv"]
            what = (f"beacon points to <b>{GOAL_NAMES[r['obs']]}</b>" if receiver == "S"
                    else f"receives clue <b>“{r['clue']}”</b>")
        dL = after - before
        k1, k2, k3 = st.columns(3)
        with k1:
            card("Decision weight w", f"{w:.2f}",
                 "min(α·π_sender, w_max)" if source == "message" else "likelihood exponent")
        with k2:
            card("Loss after", f"{after:.2f}", f"was {before:.2f} · −ln P(true)")
        with k3:
            up = dL > 1e-6
            card("Slope ΔL", f"{dL:+.2f}", ("uphill · cascade" if source == "message" else "uphill") if up
                 else "downhill", BAD if up else SENSOR)
        st.markdown(f"<div class='msg' style='margin:14px 0 4px;color:{MUTED}'>t = {t_sel} · {name} {what}"
                    f"<br>note: the ×{w_scale:.1f} multiplier reshapes the arrows; the coloured move and "
                    f"numbers are what actually happened.</div>" if abs(w_scale - 1) > 1e-9 else
                    f"<div class='msg' style='margin:14px 0 4px;color:{MUTED}'>t = {t_sel} · {name} {what}</div>",
                    unsafe_allow_html=True)
        if like is not None:
            st.markdown(f"<div class='msg' style='color:{MUTED}'>likelihood m = "
                        f"[{', '.join(f'{x:.2f}' for x in like)}] · update b ← normalize(b · m<sup>w</sup>)</div>",
                        unsafe_allow_html=True)
        st.write("")
        st.markdown("##### Descent over the episode")
        st.caption("Top: each agent's loss. Bottom: the per-step slope, split into what its own evidence "
                   "did and what messages did. Red bars are message-driven climbs — the cascade.")
        st.plotly_chart(descent_figure(trace), width="stretch", config=PLOT_CFG)

    up_ = {k: sum(max(0.0, r[f"{k}_dL_message"]) for r in trace) for k in ("S", "L")}
    dn_ = {k: -sum(min(0.0, r[f"{k}_dL_message"]) for r in trace) for k in ("S", "L")}
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        card("Sensor · loss added by messages", f"{up_['S']:.2f}", "sum of uphill steps",
             BAD if up_["S"] > 0.05 else MUTED)
    with m2:
        card("Sensor · loss removed by messages", f"{dn_['S']:.2f}", "sum of downhill steps", SENSOR)
    with m3:
        card("Language · loss added by messages", f"{up_['L']:.2f}", "sum of uphill steps",
             BAD if up_["L"] > 0.05 else MUTED)
    with m4:
        card("Language · loss removed by messages", f"{dn_['L']:.2f}", "sum of downhill steps", SENSOR)

# ─────────────────────────────────────────────────────────────────────────────
# Compare wirings on identical evidence
# ─────────────────────────────────────────────────────────────────────────────
with tab_cmp:
    st.markdown(
        "<div class='blurb'>Evidence is sampled before any messages are fused, so a fixed seed gives "
        "<b>identical</b> private observations in every condition. The only thing that changes below is "
        "who listens to whom.</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for col, m in zip(cols, MODES):
        h, sm, tr = episode(seed, steps, noise, m, flat_precision)
        verdict = ("both right" if sm["both_correct"] else "herded — agree, wrong" if sm["agree"]
                   else f"S {'✓' if sm['S_correct'] else '✗'} · L {'✓' if sm['L_correct'] else '✗'}")
        vcolor = SENSOR if sm["both_correct"] else BAD if sm["agree"] else MUTED
        with col:
            st.plotly_chart(grid_figure(h, sm["true"], height=330, animate=False, title=MODES[m]),
                            width="stretch", config=PLOT_CFG)
            up = sum(max(0.0, r["S_dL_message"]) + max(0.0, r["L_dL_message"]) for r in tr)
            st.markdown(f"<div class='msg' style='text-align:center;color:{vcolor}'>"
                        f"S → {GOAL_NAMES[sm['pred_S']]} · L → {GOAL_NAMES[sm['pred_L']]} · {verdict}</div>"
                        f"<div class='msg' style='text-align:center;color:{MUTED}'>loss added by messages: "
                        f"{up:.2f}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Noise sweep
# ─────────────────────────────────────────────────────────────────────────────
with tab_sweep:
    st.markdown("<div class='blurb'>One episode is an anecdote. This sweep runs many episodes per noise "
                "level and seed, for every wiring, and plots mean ± 1 s.d. across seeds.</div>",
                unsafe_allow_html=True)
    s1, s2, s3 = st.columns([1, 1, 2])
    n_eps = s1.select_slider("Episodes per seed", [100, 200, 400, 800], value=200)
    n_seeds = s2.select_slider("Seeds", [2, 3, 5], value=3)
    with s3:
        st.write("")
        go_sweep = st.button("Run sweep", type="primary")
    if go_sweep or st.session_state.get("sweep_ran"):
        st.session_state["sweep_ran"] = True
        with st.spinner("Simulating…"):
            df = sweep((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6), steps, n_eps, n_seeds, flat_precision)
        st.plotly_chart(sweep_figure(df), width="stretch", config=PLOT_CFG)
        st.markdown(
            "- **No communication:** the sensor holds up under noise; the language agent degrades fast.\n"
            "- **Bidirectional:** agents agree far more often, but the sensor's accuracy collapses at high "
            "noise — agreement stops meaning correctness.\n"
            "- **Unidirectional (S → L):** preserves the sensor's accuracy and lifts the weaker agent, "
            "giving the highest rate of both agents being right.")
    else:
        st.info("Press **Run sweep** (uses the timestep and precision settings from the sidebar).")

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    a, b = st.columns(2)
    with a:
        st.markdown("##### Each timestep")
        st.markdown(
            "1. **Private evidence.** The sensor sees a beacon that points to the true goal with probability "
            "1 − noise. The language agent receives a logical constraint (*“not A”*, *“either B or C”*), "
            "which is misleading with probability = noise.\n"
            "2. **Bayesian update.** Each agent multiplies its prior by the evidence likelihood and renormalises.\n"
            "3. **Message.** Beliefs are compressed into a constraint: *“not X”* if confident (max p > 0.85), "
            "otherwise *“either X or Y”*.\n"
            "4. **Fusion.** The receiver treats the message as evidence, raised to a decision weight set by "
            "the sender's precision.")
    with b:
        st.markdown("##### The maths")
        st.latex(r"b_t(g) \;\propto\; b_{t-1}(g)\,P(e_t \mid g)")
        st.latex(r"H(b) = -\sum_g b(g)\ln b(g), \qquad \pi = 1 - \frac{H(b)}{\ln 3}")
        st.latex(r"b(g) \leftarrow \frac{b(g)\,m(g)^{w}}{\sum_{g'} b(g')\,m(g')^{w}}, \qquad "
                 r"w = \min(\alpha\,\pi_{\text{sender}},\; w_{\max})")
        st.latex(r"\mathcal{L}_t = -\ln b_t(g^\star), \qquad "
                 r"\Delta\mathcal{L}_t = \Delta\mathcal{L}^{\text{evidence}}_t + \Delta\mathcal{L}^{\text{message}}_t")
        st.caption("In log space the fusion step is additive: ln b ← ln b + w · ln m (then renormalise), "
                   "so w acts as a step size on the message's log-likelihood. Soft floors (no hard zeros) "
                   "and the cap w_max keep one bad message from eliminating the truth for good.")
    st.markdown("##### Failure modes this surfaces")
    st.markdown(
        "- **Belief collapse:** an early misleading message pushes an agent into a corner it can't leave.\n"
        "- **Herding / cascade:** confident messages drive loss uphill, and agreement arrives without correctness.\n"
        "- **Miscalibrated trust:** confidence (low entropy) is not the same as reliability.")

st.markdown(f"<div style='margin-top:3rem;color:{MUTED};font-size:0.72rem;letter-spacing:0.12em;"
            f"text-transform:uppercase'>Luke Cassiano · Belief Agent · Branch A</div>", unsafe_allow_html=True)
