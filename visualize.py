#!/usr/bin/env python3
import os, re, csv
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import linregress

LOG_PATH  = Path("training.log")
RES_DIR   = Path("results")
CSV_RUNS  = RES_DIR / "run_results.csv"
CSV_RATES = RES_DIR / "run_rates.csv"
PNG_BAR   = RES_DIR / "pass_fail_bar.png"
PNG_RATE  = RES_DIR / "rate_over_runs.png"
PNG_REW   = RES_DIR / "avg_reward_over_runs.png"

RES_DIR.mkdir(exist_ok=True)

RE_NEW_RUN   = re.compile(r"=== Run (\d+) ===")
RE_SUCC_ELIM = re.compile(r"Success Rate:\s*([\d\.]+)%\s*\|\s*Elimination Rate:\s*([\d\.]+)%")
RE_EVAL_LINE = re.compile(
    r"Evaluation Reward:\s*([-+]?\d+(?:\.\d+)?)\s*\|\s*(Passed|Failed)\s*\|\s*(Eliminated faulty agent|Did not eliminate faulty agent)"
)

def parse_data(log_path: Path = LOG_PATH):
    run_results, run_rates = [], []
    cur_run_id = None
    successes, eliminations = [], []
    eval_reward = passed_flag = elim_flag = None
    def flush_run():
        nonlocal cur_run_id, successes, eliminations, eval_reward, passed_flag, elim_flag
        if cur_run_id is None or eval_reward is None:
            return
        avg_success = np.mean(successes) if successes else 0.0
        avg_elim    = np.mean(eliminations) if eliminations else 0.0
        run_results.append(dict(
            Run=cur_run_id,
            Reward=eval_reward,
            successfulCompletion=int(passed_flag),
            successfulElimination=int(elim_flag)
        ))
        run_rates.append(dict(
            Run=cur_run_id,
            AvgSuccessRate=round(avg_success, 3),
            AvgEliminationRate=round(avg_elim, 3)
        ))
        cur_run_id = None
        successes, eliminations = [], []
        eval_reward = passed_flag = elim_flag = None
    with log_path.open("rb") as fb:
        for raw in fb:
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                line = raw.decode("latin1", "ignore")
            m_run = RE_NEW_RUN.search(line)
            if m_run:
                flush_run()
                cur_run_id = int(m_run.group(1))
                continue
            m_rate = RE_SUCC_ELIM.search(line)
            if m_rate:
                successes.append(float(m_rate.group(1)))
                eliminations.append(float(m_rate.group(2)))
                continue
            m_eval = RE_EVAL_LINE.search(line)
            if m_eval:
                eval_reward = float(m_eval.group(1))
                passed_flag = m_eval.group(2) == "Passed"
                elim_flag   = m_eval.group(3).startswith("Eliminated")
    flush_run()
    df_runs  = pd.DataFrame(run_results).sort_values("Run").reset_index(drop=True)
    df_rates = pd.DataFrame(run_rates).sort_values("Run").reset_index(drop=True)
    df_runs.to_csv(CSV_RUNS,  index=False)
    df_rates.to_csv(CSV_RATES, index=False)
    print(f"Parsed {len(df_runs)} runs → {CSV_RUNS.name}, {CSV_RATES.name}")
    return df_runs, df_rates

def graph_bar_chart(df_runs: pd.DataFrame, out_path: Path = PNG_BAR):
    total_runs = len(df_runs)
    pass_cnt   = df_runs["successfulCompletion"].sum()
    fail_cnt   = total_runs - pass_cnt
    elim_pass  = df_runs["successfulElimination"].sum()
    elim_fail  = total_runs - elim_pass
    labels   = ["Run Failed", "Run Passed", "Elim Failed", "Elim Passed"]
    counts   = [fail_cnt, pass_cnt, elim_fail, elim_pass]
    colors   = ["#377eb8", "#ff7f0e", "#377eb8", "#ff7f0e"]
    x_pos    = [0, 1, 3, 4]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x_pos, counts, color=colors, width=0.8)
    ax.set_xticks(x_pos, labels)
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_title("Overall Outcome Counts")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, cnt + 0.5, f"{cnt}", ha="center", va="bottom", fontsize=9)
    ax.text(0.99, 1.04, f"Total runs: {total_runs}", transform=ax.transAxes, ha="right", va="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart  → {out_path}")

def _scatter_line(ax, x, y, ylabel, color):
    ax.scatter(x, y, color=color, alpha=0.8)
    slope, intercept, *_ = linregress(x, y)
    ax.plot(x, slope * x + intercept, color=color, linewidth=2, linestyle="--")
    for xi in x:
        ax.axvline(xi, color="gray", linestyle=":", alpha=0.15, linewidth=0.8)
    ax.set_ylabel(ylabel)
    avg_val = y.mean()
    ax.text(0.02, 0.9, f"Average = {avg_val:.1f}%", transform=ax.transAxes, ha="left", va="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))

def graph_rate_run(df_rates: pd.DataFrame, out_path: Path = PNG_RATE):
    x = df_rates["Run"].values
    succ = df_rates["AvgSuccessRate"].values
    elim = df_rates["AvgEliminationRate"].values
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={"hspace": 0.3})
    _scatter_line(ax1, x, succ, "Success Rate (%)", "#2b8cbe")
    _scatter_line(ax2, x, elim, "Elimination Rate (%)", "#e6550d")
    ax2.set_xlabel("Run")
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    fig.suptitle("Rates over Runs")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Rate plots → {out_path}")

def graph_reward_run(df_runs: pd.DataFrame, out_path: Path = PNG_REW):
    x = df_runs["Run"].values
    y = df_runs["Reward"].values
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(x, y, color="#4daf4a", alpha=0.8)
    slope, intercept, *_ = linregress(x, y)
    ax.plot(x, slope * x + intercept, color="#4daf4a", linewidth=2, linestyle="--")
    for xi in x:
        ax.axvline(xi, color="gray", linestyle=":", alpha=0.15, linewidth=0.8)
    ax.set_xlabel("Run")
    ax.set_ylabel("Evaluation Reward")
    ax.set_title("Average Reward over Runs")
    avg_val = y.mean()
    ax.text(0.02, 0.9, f"Average = {avg_val:.2f}", transform=ax.transAxes, ha="left", va="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Reward plot → {out_path}")

def main():
    df_runs, df_rates = parse_data()
    graph_bar_chart(df_runs)
    graph_rate_run(df_rates)
    graph_reward_run(df_runs)

if __name__ == "__main__":
    main()
