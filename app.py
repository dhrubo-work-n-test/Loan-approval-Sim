import streamlit as st
import simpy
import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import product
import time
import io

# ---------------------------
# Streamlit page config + style
# ---------------------------
st.set_page_config(page_title="Loan Approval Simulator", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #071032 100%); color: #e6eef8; }
    .big-font { font-size:22px !important; font-weight:600; }
    .muted { color: #a8b3c7; }
    .card { background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<h1 style='margin:8px 0 4px'>🏦 Loan Approval Flow — Interactive Simulator</h1>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Visualize queues, find bottlenecks, optimize staffing under budget, and export results.</div>", unsafe_allow_html=True)
st.write("")

# ---------------------------
# Defaults & Model
# ---------------------------
DEFAULT_STAGES = [
    "Document Collection",
    "Initial Review",
    "Credit Check",
    "Underwriting",
    "Final Approval",
]

DEFAULT_PROC = {
    "Document Collection": 10.0,
    "Initial Review": 20.0,
    "Credit Check": 30.0,
    "Underwriting": 25.0,
    "Final Approval": 10.0,
}

DEFAULT_STAFF = {
    "Document Collection": 3,
    "Initial Review": 2,
    "Credit Check": 1,
    "Underwriting": 4,
    "Final Approval": 1,
}

# ---------------------------
# Sidebar - controls
# ---------------------------
st.sidebar.header("🛠 Simulation Controls")

num_loans = st.sidebar.number_input("Loan applications (total)", min_value=10, max_value=5000, value=500, step=10)
days = st.sidebar.number_input("Simulate (working days, 8h/day)", min_value=1, max_value=30, value=1)
arrival_interval = st.sidebar.slider("Arrival mean (minutes)", min_value=0.5, max_value=10.0, value=1.0)

st.sidebar.markdown("---")
st.sidebar.header("👥 Staff (per stage)")
staff = {}
for s in DEFAULT_STAGES:
    staff[s] = st.sidebar.slider(s, 0, 30, DEFAULT_STAFF.get(s, 1), 1)

st.sidebar.markdown("---")
st.sidebar.header("💸 Cost & Optimization")
cost_per_staff = st.sidebar.number_input("Cost per staff / day (₹)", min_value=0, max_value=50000, value=2000, step=100)
budget = st.sidebar.number_input("Budget available / day (₹)", min_value=0, max_value=1000000, value=20000, step=500)
optimize_btn = st.sidebar.button("🔎 Optimize allocation (within budget)")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with **SimPy** (discrete-event) + **Streamlit** • Tip: reduce search window when optimizing to speed up runs.")

# ---------------------------
# Simulation engine (SimPy)
# ---------------------------
class LoanSystem:
    def __init__(self, env, stages, processing_times, staff_counts):
        self.env = env
        self.stages = stages
        self.processing_times = processing_times
        # create resources (capacity = staff count, min 1 to avoid zero-cap Resource)
        self.resources = {s: simpy.Resource(env, capacity=max(1, staff_counts.get(s, 1))) for s in stages}
        self.stage_queue_time = {s: [] for s in stages}
        self.stage_queue_length_t = {s: [] for s in stages}
        self.stage_completed = {s: 0 for s in stages}
        self.total_times = []

    def process(self, name):
        start = self.env.now
        for s in self.stages:
            arrival = self.env.now
            res = self.resources[s]
            # record queue length snapshot (before requesting)
            self.stage_queue_length_t[s].append((self.env.now, len(res.queue)))
            with res.request() as req:
                yield req
                wait = self.env.now - arrival
                self.stage_queue_time[s].append(wait)
                # processing time (minutes)
                yield self.env.timeout(self.processing_times[s])
                self.stage_completed[s] += 1
        self.total_times.append(self.env.now - start)

def run_simulation(stages, processing_times, staff_counts, n_loans, arrival_mean, days):
    """
    Runs the sim for `days` working days (8 hours = 480 minutes per day).
    Returns: system object, df_queues dict, summary dict
    """
    minutes_per_day = 480
    sim_time = minutes_per_day * days

    env = simpy.Environment()
    system = LoanSystem(env, stages, processing_times, staff_counts)

    def arrival_gen(env):
        i = 0
        while env.now < sim_time and i < n_loans:
            env.process(system.process(f"loan_{i}"))
            i += 1
            # exponential inter-arrival
            inter = random.expovariate(1.0 / arrival_mean)
            yield env.timeout(inter)

    env.process(arrival_gen(env))

    def sampler(env):
        while env.now < sim_time:
            for s, r in system.resources.items():
                system.stage_queue_length_t[s].append((env.now, len(r.queue)))
            yield env.timeout(1)  # sample each minute

    env.process(sampler(env))
    env.run(until=sim_time)

    # build queue dataframes
    df_queues = {}
    for s, rec in system.stage_queue_length_t.items():
        if rec:
            times, lens = zip(*rec)
            df_queues[s] = pd.DataFrame({"time_min": times, "queue_len": lens})
        else:
            df_queues[s] = pd.DataFrame(columns=["time_min", "queue_len"])

    summary = {
        "avg_total_time_min": float(np.mean(system.total_times)) if system.total_times else float("nan"),
        "median_total_time_min": float(np.median(system.total_times)) if system.total_times else float("nan"),
        "completed": len(system.total_times),
        "stage_completed": system.stage_completed,
        "stage_avg_wait_min": {s: float(np.mean(system.stage_queue_time[s])) if system.stage_queue_time[s] else 0.0 for s in stages},
        "stage_avg_proc_min": {s: processing_times[s] for s in stages},
    }

    return system, df_queues, summary

# ---------------------------
# UI: Run simulation
# ---------------------------
col_l, col_r = st.columns([1, 2], gap="large")

with col_l:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧭 Configuration")
    st.write("Tweak arrivals, staff, and cost parameters in the left sidebar. Then run the simulation.")
    if st.button("▶ Run Simulation", key="run_sim"):
        t0 = time.time()
        system, df_queues, summary = run_simulation(DEFAULT_STAGES, DEFAULT_PROC, staff, num_loans, arrival_interval, days)
        st.session_state["result"] = (system, df_queues, summary, dict(staff))  # store staff snapshot
        t1 = time.time()
        st.success(f"Simulation finished — {summary['completed']} loans processed (sim time: {days} day(s)). ({t1-t0:.1f}s)")
    st.markdown("</div>", unsafe_allow_html=True)

with col_r:
    st.subheader("📈 Visuals & Metrics")
    if "result" in st.session_state:
        system, df_queues, summary, staff_snapshot = st.session_state["result"]

        # Top KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Avg approval time (min)", f"{summary['avg_total_time_min']:.1f}")
        k2.metric("Median approval time (min)", f"{summary['median_total_time_min']:.1f}")
        k3.metric("Completed loans (sim)", f"{summary['completed']}")

        # Bottleneck bar (avg wait)
        wait_df = pd.DataFrame(summary["stage_avg_wait_min"], index=["avg_wait_min"]).T
        wait_df = wait_df.sort_values("avg_wait_min", ascending=False)
        fig_wait = px.bar(wait_df, x=wait_df.index, y="avg_wait_min", title="Average Wait Time by Stage (mins)")
        fig_wait.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig_wait, use_container_width=True)

        # Queue length over time (stacked area)
        qfig = go.Figure()
        for s, dfq in df_queues.items():
            if not dfq.empty:
                agg = dfq.groupby("time_min").queue_len.mean().reset_index()
                qfig.add_trace(go.Scatter(x=agg.time_min, y=agg.queue_len, mode="lines", stackgroup="one", name=s, hoverinfo="x+y+name"))
        qfig.update_layout(title="Queue Pressure Over Time", xaxis_title="Time (min)", yaxis_title="Queue size (avg)", showlegend=True, margin=dict(t=40))
        st.plotly_chart(qfig, use_container_width=True)

        # Workload distribution donut (Style 3: exploding max slice)
        # Compute workload contribution = processing_time * completed_count (approx work-minutes)
        labels = DEFAULT_STAGES
        completed_counts = [system.stage_completed[s] for s in labels]
        proc_times = [DEFAULT_PROC[s] for s in labels]
        workload = [c * p for c, p in zip(completed_counts, proc_times)]
        total_work = sum(workload) if sum(workload) > 0 else 1
        pct = [100.0 * w / total_work for w in workload]

        # determine explode/pull for max slice
        if len(pct) > 0:
            max_idx = int(np.argmax(pct))
            pulls = [0.0] * len(pct)
            pulls[max_idx] = 0.12  # explode the max slice
        else:
            pulls = [0.0] * len(pct)

        donut = go.Figure(data=[go.Pie(labels=labels,
                                       values=workload,
                                       hole=0.55,
                                       pull=pulls,
                                       hoverinfo="label+percent+value",
                                       textinfo="percent+label")])
        donut.update_layout(title="Workload Distribution (work-minutes) — bottleneck highlighted", margin=dict(t=40))
        st.plotly_chart(donut, use_container_width=True)

        # Table: detailed stage stats
        st.markdown("**Stage details**")
        detail_df = pd.DataFrame({
            "stage": labels,
            "completed": completed_counts,
            "avg_wait_min": [summary["stage_avg_wait_min"][s] for s in labels],
            "proc_time_min": [DEFAULT_PROC[s] for s in labels],
            "work_minutes": workload,
            "work_pct": [f"{p:.1f}%" for p in pct],
            "staff": [staff_snapshot[s] for s in labels],
        }).set_index("stage")
        st.dataframe(detail_df.style.format({"avg_wait_min": "{:.2f}", "proc_time_min": "{:.1f}", "work_minutes": "{:.1f}"}))
    else:
        st.info("Run the simulation to see visuals and metrics.")

# ---------------------------
# Export section
# ---------------------------
st.markdown("---")
st.subheader("📤 Export / Download")

if "result" in st.session_state:
    system, df_queues, summary, staff_snapshot = st.session_state["result"]

    df_summary = pd.DataFrame({
        "stage": DEFAULT_STAGES,
        "staff": [staff_snapshot[s] for s in DEFAULT_STAGES],
        "completed": [system.stage_completed[s] for s in DEFAULT_STAGES],
        "avg_wait_min": [summary["stage_avg_wait_min"][s] for s in DEFAULT_STAGES],
        "proc_time_min": [DEFAULT_PROC[s] for s in DEFAULT_STAGES],
        "work_minutes": [system.stage_completed[s] * DEFAULT_PROC[s] for s in DEFAULT_STAGES],
    }).set_index("stage")

    csv = df_summary.to_csv().encode("utf-8")
    st.download_button("Download stage summary (CSV)", csv, "stage_summary.csv", mime="text/csv")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df_summary.to_excel(writer, sheet_name="stage_summary")
    st.download_button("Download stage summary (Excel)", excel_buffer.getvalue(),
                       "stage_summary.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # export queue time series as zipped csvs (lightweight)
    zip_buffer = io.BytesIO()
    import zipfile
    with zipfile.ZipFile(zip_buffer, mode="w") as zf:
        for s, dfq in df_queues.items():
            csv_bytes = dfq.to_csv(index=False).encode("utf-8")
            zf.writestr(f"queue_{s.replace(' ', '_')}.csv", csv_bytes)
    st.download_button("Download queue timeseries (ZIP)", zip_buffer.getvalue(), "queue_timeseries.zip", mime="application/zip")
else:
    st.info("Run simulation to enable export options.")

# ---------------------------
# Optimization: find best staff allocation under budget
# ---------------------------
st.markdown("---")
st.header("🧠 Optimization — Allocate Staff Under Budget")

st.write("This performs a small grid search around current staff levels (±2) to find a configuration that minimizes average approval time while respecting daily staff cost budget.")
st.write("Note: this is an experimental search — narrow the staff sliders or reduce ranges to speed it up.")

if optimize_btn:
    if "result" not in st.session_state:
        st.warning("Run a simulation first so optimizer has baseline context.")
    else:
        # construct local search ranges (bounded)
        current_staff = staff.copy()
        ranges = {}
        for s, c in current_staff.items():
            low = max(0, c - 2)
            high = c + 2
            ranges[s] = range(low, high + 1)

        # compute candidate count
        candidate_count = np.prod([len(r) for r in ranges.values()])
        st.info(f"Searching {candidate_count} configurations (this may take some time).")
        progress = st.progress(0)

        best_metric = None
        best_conf = None
        tried = 0

        # iterate candidates
        for idx, combo in enumerate(product(*ranges.values())):
            candidate = dict(zip(ranges.keys(), combo))
            total_staff_cost = sum(candidate[s] * cost_per_staff for s in candidate)
            if total_staff_cost > budget:
                tried += 1
                progress.progress(int(tried / candidate_count * 100))
                continue
            # run sim (small sim: keep same n_loans/days)
            system_c, dfq_c, summary_c = run_simulation(DEFAULT_STAGES, DEFAULT_PROC, candidate, num_loans, arrival_interval, days)
            metric = summary_c["avg_total_time_min"]
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_conf = (candidate, total_staff_cost, summary_c)
            tried += 1
            progress.progress(int(tried / candidate_count * 100))

        if best_conf:
            cand, tot_cost, summ = best_conf
            st.success("Optimization complete — best config found (within search window).")
            st.subheader("Recommended staff allocation")
            st.write(pd.DataFrame.from_dict(cand, orient="index", columns=["staff"]))
            st.write(f"Total staff cost/day: ₹{tot_cost}")
            st.write(f"Expected avg approval time (min): {summ['avg_total_time_min']:.1f}")
            st.session_state["opt_result"] = best_conf
        else:
            st.error("No feasible configuration found within the budget and search window.")

# show opt result (if any)
if "opt_result" in st.session_state:
    st.markdown("### Best configuration found earlier")
    cand, tot_cost, summ = st.session_state["opt_result"]
    st.write(pd.DataFrame.from_dict(cand, orient="index", columns=["staff"]))
    st.write(f"Cost/day: ₹{tot_cost} • Avg approval time: {summ['avg_total_time_min']:.1f} min")

# ---------------------------
# Footer / notes
# ---------------------------
st.markdown("---")
st.caption("Tip: For production use add persistent storage, longer-lived arrival traces, cost tradeoff curves, and faster optimizers (ILP / Bayesian).")
