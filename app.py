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

# ----------------------------------------------------
# Streamlit Config
# ----------------------------------------------------
st.set_page_config(page_title="Loan Approval Simulation", layout="wide")
st.title("🏦 Loan Approval Flow Simulator")
st.write("Simulate loan approvals, visualize bottlenecks, run optimizations, and explore cost vs speed trade-offs.")

# ----------------------------------------------------
# Default Model Definitions
# ----------------------------------------------------
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

default_staff = {
    "Document Collection": 3,
    "Initial Review": 2,
    "Credit Check": 1,
    "Underwriting": 4,
    "Final Approval": 1,
}

# ----------------------------------------------------
# Sidebar Inputs
# ----------------------------------------------------
st.sidebar.header("Simulation Controls")

num_loans = st.sidebar.number_input("Number of loan applications", 10, 2000, 500, 10)
days = st.sidebar.number_input("Simulate (days)", 1, 30, 1)
arrival_interval = st.sidebar.slider("Inter-arrival mean (minutes)", 0.5, 10.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.header("Staff Configuration (per stage)")

staff = {}
for stage in DEFAULT_STAGES:
    staff[stage] = st.sidebar.slider(stage, 0, 20, default_staff.get(stage, 1))

st.sidebar.markdown("---")
st.sidebar.header("Cost & Optimization")
cost_per_staff = st.sidebar.number_input("Cost per staff per day (₹)", 0, 10000, 2000)
budget = st.sidebar.number_input("Max staff budget per day (₹)", 0, 200000, 20000)

optimize_btn = st.sidebar.button("🔎 Optimize Allocation Within Budget")

# ----------------------------------------------------
# Simulation Engine
# ----------------------------------------------------
class LoanSystem:
    def __init__(self, env, stages, processing_times, staff_counts):
        self.env = env
        self.stages = stages
        self.processing_times = processing_times
        self.resources = {
            s: simpy.Resource(env, capacity=staff_counts.get(s, 1)) 
            for s in stages
        }
        self.stage_queue_time = {s: [] for s in stages}
        self.stage_queue_length_t = {s: [] for s in stages}
        self.stage_completed = {s: 0 for s in stages}
        self.total_times = []

    def process(self, name):
        start = self.env.now
        for s in self.stages:
            arrival = self.env.now
            res = self.resources[s]

            self.stage_queue_length_t[s].append((self.env.now, len(res.queue)))

            with res.request() as req:
                yield req
                wait = self.env.now - arrival
                self.stage_queue_time[s].append(wait)

                yield self.env.timeout(self.processing_times[s])
                self.stage_completed[s] += 1

        self.total_times.append(self.env.now - start)


def run_simulation(stages, processing_times, staff_counts, n_loans, arrival_mean, days):
    minutes_per_day = 480
    sim_time = minutes_per_day * days

    env = simpy.Environment()
    system = LoanSystem(env, stages, processing_times, staff_counts)

    def generator(env):
        i = 0
        while env.now < sim_time and i < n_loans:
            env.process(system.process(f"loan_{i}"))
            i += 1
            inter = random.expovariate(1.0 / arrival_mean)
            yield env.timeout(inter)

    env.process(generator(env))

    def sampler(env):
        while env.now < sim_time:
            for s, r in system.resources.items():
                system.stage_queue_length_t[s].append((env.now, len(r.queue)))
            yield env.timeout(1)

    env.process(sampler(env))

    env.run(until=sim_time)

    df_queues = {}
    for s, rec in system.stage_queue_length_t.items():
        if rec:
            t, l = zip(*rec)
            df_queues[s] = pd.DataFrame({"time_min": t, "queue_len": l})
        else:
            df_queues[s] = pd.DataFrame(columns=["time_min", "queue_len"])

    summary = {
        "avg_total_time_min": float(np.mean(system.total_times)) if system.total_times else float("nan"),
        "median_total_time_min": float(np.median(system.total_times)) if system.total_times else float("nan"),
        "completed": len(system.total_times),
        "stage_completed": system.stage_completed,
        "stage_avg_wait_min": {s: float(np.mean(system.stage_queue_time[s])) if system.stage_queue_time[s] else 0
                               for s in system.stages},
    }

    return system, df_queues, summary

# ----------------------------------------------------
# UI — Run Simulation
# ----------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Staff Configuration")
    st.write(pd.DataFrame([staff]).T.rename(columns={0: "Staff"}))

    if st.button("▶ Run Simulation"):
        system, df_queues, summary = run_simulation(
            DEFAULT_STAGES, DEFAULT_PROC, staff, num_loans, arrival_interval, days
        )
        st.session_state["result"] = (system, df_queues, summary)
        st.success("Simulation Completed!")

with col2:
    st.subheader("Visualizations")

    if "result" in st.session_state:
        system, df_queues, summary = st.session_state["result"]

        wait_df = pd.DataFrame(summary["stage_avg_wait_min"], index=["avg_wait_min"]).T
        wait_df = wait_df.sort_values("avg_wait_min", ascending=False)

        fig1 = px.bar(wait_df, x=wait_df.index, y="avg_wait_min", title="Bottlenecks — Average Wait Time (mins)")
        st.plotly_chart(fig1, use_container_width=True)

        qfig = go.Figure()
        for s, dfq in df_queues.items():
            if not dfq.empty:
                agg = dfq.groupby("time_min").queue_len.mean().reset_index()
                qfig.add_trace(go.Scatter(x=agg.time_min, y=agg.queue_len, mode="lines", name=s))
        qfig.update_layout(title="Queue Length Over Time", xaxis_title="Minutes", yaxis_title="Queue Size")
        st.plotly_chart(qfig, use_container_width=True)

        labels = DEFAULT_STAGES
        cnt = [system.stage_completed[s] for s in labels]
        sources, targets, values = [], [], []
        for i in range(len(labels) - 1):
            sources.append(i)
            targets.append(i + 1)
            values.append(cnt[i + 1])
        sankey = go.Figure(data=[go.Sankey(node={"label": labels},
                                           link={"source": sources, "target": targets, "value": values})])
        st.plotly_chart(sankey, use_container_width=True)

        st.subheader("Summary")
        st.metric("Average Approval Time (min)", f"{summary['avg_total_time_min']:.1f}")
        st.metric("Median Approval Time (min)", f"{summary['median_total_time_min']:.1f}")
        st.write(pd.DataFrame(summary["stage_avg_wait_min"], index=["avg_wait_min"]).T)


# ----------------------------------------------------
# Export / Helpers
# ----------------------------------------------------
if "result" in st.session_state:
    st.markdown("---")
    st.subheader("📤 Export Results")

    system, df_queues, summary = st.session_state["result"]

    df_summary = pd.DataFrame(summary["stage_avg_wait_min"], index=["avg_wait_min"]).T
    df_summary["completed"] = list(summary["stage_completed"].values())

    csv_data = df_summary.to_csv().encode("utf-8")
    st.download_button("Download Summary CSV", csv_data, "simulation_summary.csv")

    json_data = pd.Series(summary).to_json().encode("utf-8")
    st.download_button("Download Summary JSON", json_data, "simulation_summary.json")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df_summary.to_excel(writer, index=True)
    st.download_button("Download Summary Excel", excel_buffer.getvalue(),
                       "simulation_summary.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------------------------------------------
# Optimization
# ----------------------------------------------------
st.markdown("---")
st.header("Optimization — Find Best Configuration Under Budget")

if optimize_btn:
    st.info("Searching for optimal staff configuration...")

    ranges = {}
    for s, c in staff.items():
        ranges[s] = range(max(0, c - 2), c + 3)

    best = None
    best_conf = None
    total_candidates = np.prod([len(r) for r in ranges.values()])
    progress = st.progress(0)

    for i, combo in enumerate(product(*ranges.values())):
        candidate = dict(zip(ranges.keys(), combo))
        total_cost = sum(candidate[s] * cost_per_staff for s in candidate)

        if total_cost > budget:
            continue

        system, df_queues, summary = run_simulation(
            DEFAULT_STAGES, DEFAULT_PROC, candidate, num_loans, arrival_interval, days
        )

        metric = summary["avg_total_time_min"]
        if best is None or metric < best:
            best = metric
            best_conf = (candidate, total_cost, summary)

        progress.progress(int((i + 1) / total_candidates * 100))

    if best_conf:
        st.success("Optimization Complete!")
        cand, tot_cost, summ = best_conf

        st.subheader("Best Staff Allocation")
        st.write(pd.DataFrame.from_dict(cand, orient="index", columns=["staff"]))
        st.write(f"Total Cost/day: ₹{tot_cost}")
        st.write(f"Expected Avg Approval Time: {summ['avg_total_time_min']:.1f} min")
    else:
        st.error("No feasible configuration found within budget.")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.caption("Loan Approval Simulator • Streamlit + SimPy • Built for Capacity Planning & Bottleneck Analysis")
