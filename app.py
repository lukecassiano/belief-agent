import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

GOAL_POS = {
    "A": (0.0, 1.0),
    "B": (-1.0, -1.0),
    "C": (1.0, -1.0),
}

def belief_to_xy(belief):
    # belief is np.array([pA, pB, pC])
    Ax, Ay = GOAL_POS["A"]
    Bx, By = GOAL_POS["B"]
    Cx, Cy = GOAL_POS["C"]
    pA, pB, pC = float(belief[0]), float(belief[1]), float(belief[2])
    x = pA * Ax + pB * Bx + pC * Cx
    y = pA * Ay + pB * By + pC * Cy
    return x, y

def draw_grid_frame(step, trail_S, trail_L):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Draw "grid" (light reference)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    # Plot goals
    for g, (x, y) in GOAL_POS.items():
        ax.scatter([x], [y], s=200)
        ax.text(x, y + 0.08, g, ha="center", va="bottom", fontsize=14)

    # Trails
    if len(trail_S) > 1:
        xs, ys = zip(*trail_S)
        ax.plot(xs, ys, linewidth=2, alpha=0.8)
    if len(trail_L) > 1:
        xl, yl = zip(*trail_L)
        ax.plot(xl, yl, linewidth=2, alpha=0.8)

    # Current positions
    sx, sy = trail_S[-1]
    lx, ly = trail_L[-1]
    ax.scatter([sx], [sy], s=160, marker="o", label="Sensor")
    ax.scatter([lx], [ly], s=160, marker="s", label="Language")

    # Title / annotations
    ax.set_title(f"t={step['t']} | true={['A','B','C'][step['true_goal']]} | obs={['A','B','C'][step['obs']]}")

    # Put messages under plot
    msg = f"S msg: {step['S_msg']} | L msg: {step['L_msg']} | clue: {step['clue']}"
    ax.text(0.5, -0.10, msg, transform=ax.transAxes, ha="center", va="top", fontsize=9)

    ax.legend(loc="upper right")
    return fig


from simulation import (
    GOAL_NAMES,
    run_episode_history,
    run_many_mode,
)


st.set_page_config(page_title="Uncertainty-Aware Multi-Agent Inference", layout="wide")
st.title("Uncertainty-Aware Multi-Agent Inference — Branch A UI")




# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Controls")
    noise = st.slider("Noise", 0.0, 0.6, 0.30, 0.05)
    steps = st.slider("Steps", 1, 12, 6, 1)
    episodes = st.number_input("Episodes (for sweeps)", value=1000, step=100)
    seeds = st.number_input("Seeds (for sweeps)", value=5, step=1, min_value=1)

    mode = st.selectbox("Comm mode (episode tab)", ["none", "bidirectional", "unidirectional_S_to_L"], index=1)

    precision_mode = st.selectbox("Precision weighting", ["dynamic", "flat"], index=0)
    flat_p = st.slider("Flat precision p", 0.0, 1.0, 0.5, 0.05)

    seed = st.number_input("Seed (episode tab)", value=42, step=1)


tab1, tab2 = st.tabs(["Episode Visualizer", "Noise Sweep"])


# =========================
# Tab 1: Episode Visualizer
# =========================
with tab1:
    st.subheader("Episode Visualizer")

    if "hist" not in st.session_state:
        st.session_state.hist = None
        st.session_state.summary = None

    if st.button("Run episode"):
        rng = np.random.default_rng(int(seed))
        flat_precision = None if precision_mode == "dynamic" else float(flat_p)
        hist, summary = run_episode_history(
            rng=rng,
            steps=int(steps),
            noise=float(noise),
            mode=mode,
            flat_precision=flat_precision,
        )
        st.session_state.hist = hist
        st.session_state.summary = summary

    hist = st.session_state.hist
    summary = st.session_state.summary

    if hist is None:
        st.info("Set controls and click **Run episode**.")
    else:
        true_goal = GOAL_NAMES[summary["true"]]
        pred_S = GOAL_NAMES[summary["pred_S"]]
        pred_L = GOAL_NAMES[summary["pred_L"]]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True goal", true_goal)
        c2.metric("Sensor prediction", pred_S, "✅" if summary["S_correct"] else "❌")
        c3.metric("Language prediction", pred_L, "✅" if summary["L_correct"] else "❌")
        c4.metric("Agree?", "Yes" if summary["agree"] else "No", "✅" if summary["both_correct"] else "")

        idx = st.slider("Step", 1, len(hist), 1)
        step = hist[idx - 1]

        def belief_df(b):
            return pd.DataFrame({"goal": ["A", "B", "C"], "p": b}).set_index("goal")

        left, right = st.columns(2)
        with left:
            st.markdown("### Sensor agent")
            st.write(f"**Obs:** points to **{GOAL_NAMES[step['obs']]}**")
            st.write(f"**Message:** `{step['S_msg']}`")
            st.write(f"Entropy: `{step['S_entropy']:.3f}` | Precision: `{step['S_precision']:.2f}`")
            st.bar_chart(belief_df(step["S_belief"]))

        with right:
            st.markdown("### Language agent")
            st.write(f"**Clue:** **{step['clue']}**")
            st.write(f"**Message:** `{step['L_msg']}`")
            st.write(f"Entropy: `{step['L_entropy']:.3f}` | Precision: `{step['L_precision']:.2f}`")
            st.bar_chart(belief_df(step["L_belief"]))

        st.divider()
        st.subheader("Full log")
        rows = []
        for stp in hist:
            rows.append({
                "t": stp["t"],
                "obs": GOAL_NAMES[stp["obs"]],
                "clue": stp["clue"],
                "S_msg": stp["S_msg"],
                "L_msg": stp["L_msg"],
                "S_precision": round(stp["S_precision"], 2),
                "L_precision": round(stp["L_precision"], 2),
                "S_entropy": round(stp["S_entropy"], 3),
                "L_entropy": round(stp["L_entropy"], 3),
                "S_A": round(float(stp["S_belief"][0]), 3),
                "S_B": round(float(stp["S_belief"][1]), 3),
                "S_C": round(float(stp["S_belief"][2]), 3),
                "L_A": round(float(stp["L_belief"][0]), 3),
                "L_B": round(float(stp["L_belief"][1]), 3),
                "L_C": round(float(stp["L_belief"][2]), 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.subheader("Mapped grid animation (belief → position)")

animate = st.button("Animate episode on grid")
speed = st.slider("Animation speed (seconds per step)", 0.1, 1.5, 0.6, 0.1)
show_trails = st.checkbox("Show trails", value=True)

if animate and hist is not None:
    placeholder = st.empty()

    trail_S = []
    trail_L = []

    for step in hist:
        sxy = belief_to_xy(step["S_belief"])
        lxy = belief_to_xy(step["L_belief"])
        trail_S.append(sxy)
        trail_L.append(lxy)

        if not show_trails:
            trail_S = [trail_S[-1]]
            trail_L = [trail_L[-1]]

        fig = draw_grid_frame(step, trail_S, trail_L)
        placeholder.pyplot(fig, clear_figure=True)
        plt.close(fig)
        time.sleep(float(speed))


# =================
# Tab 2: Noise Sweep
# =================
with tab2:
    st.subheader("Noise Sweep")

    st.write("Runs **no comm**, **bidirectional**, and **unidirectional (S→L)** across noise values and plots metrics.")

    sweep_noise_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    metric = st.selectbox("Metric to plot", ["S_correct", "L_correct", "both_correct", "agree"], index=1)

    run_sweep = st.button("Run sweep")

    if "sweep_df" not in st.session_state:
        st.session_state.sweep_df = None

    if run_sweep:
        flat_precision = None if precision_mode == "dynamic" else float(flat_p)

        modes = {
            "no_comm": "none",
            "bidirectional": "bidirectional",
            "unidirectional_S_to_L": "unidirectional_S_to_L",
        }

        records = []
        for nz in sweep_noise_values:
            for label, m in modes.items():
                per_seed = []
                for s in range(int(seeds)):
                    stats = run_many_mode(
                        seed=1000 + s,
                        episodes=int(episodes),
                        steps=int(steps),
                        noise=float(nz),
                        mode=m,
                        flat_precision=flat_precision,
                    )
                    per_seed.append(stats)

                mean = float(np.mean([d[metric] for d in per_seed]))
                std = float(np.std([d[metric] for d in per_seed]))
                records.append({
                    "noise": nz,
                    "mode": label,
                    f"{metric}_mean": mean,
                    f"{metric}_std": std
                })

        df = pd.DataFrame(records)
        st.session_state.sweep_df = df

    df = st.session_state.sweep_df
    if df is None:
        st.info("Click **Run sweep** to generate plots.")
    else:
        st.write("Means across seeds (std shown in table).")
        pivot = df.pivot(index="noise", columns="mode", values=f"{metric}_mean").sort_index()
        st.line_chart(pivot)

        st.divider()
        st.subheader("Sweep table")
        st.dataframe(df.sort_values(["noise", "mode"]), use_container_width=True)

