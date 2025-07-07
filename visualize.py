import os
import re
import statistics
from collections import deque
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LOG_PATH        = "training.log"
RES_DIR         = "results"
RUN_RESULTS_CSV = os.path.join(RES_DIR, "run_results.csv")
RUN_RATES_CSV   = os.path.join(RES_DIR, "run_rates.csv")
os.makedirs(RES_DIR, exist_ok=True)

RE_RUN_HDR     = re.compile(r"=== Run (\d+) ===")
RE_TRAIN_START = re.compile(r"Starting training for\s+(\d+)\s+episode")

RE_TRAIN_AVG   = re.compile(
    r"Avg Reward:\s*([-\d.]+)\s*\|\s*Success Rate:\s*([\d.]+)%\s*\|\s*Elimination Rate:\s*([\d.]+)%"
)

RE_EVAL_NUM = re.compile(
    r"Eval.*mean reward\s+([-\d.]+)\s*\|\s*Success\s+([\d.]+)%?\s*\|\s*Elim\s+([\d.]+)%?",
    re.IGNORECASE,
)
RE_EVAL_YN  = re.compile(
    r"Eval(?:uation)?[^-0-9]*mean reward\s+([-\d.]+)\s*\|\s*Success\s+([YN])\s*\|\s*Elim(?:ination)?\s+([YN])",
    re.IGNORECASE,
)
RE_LOSS = re.compile(r"\bLoss\s+([-\d.]+)")

def parse_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_results, run_rates, run_losses = [], [], []

    cur_run        = None
    cur_episodes   = 0           
    global_counter = 0          
    loss_buf: List[float] = []

    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            # ── new-run header ─────────────────────────────
            if (m_run := RE_RUN_HDR.search(line)):
                # flush loss buffer from the previous run
                if cur_run is not None and loss_buf:
                    run_losses.append([global_counter, np.mean(loss_buf)])
                    loss_buf.clear()
                # advance episode counter to start of new run
                global_counter += cur_episodes
                cur_run      = int(m_run.group(1))
                cur_episodes = 0
                continue

            if (m_start := RE_TRAIN_START.search(line)):
                cur_episodes = int(m_start.group(1))
                continue

            if (m_avg := RE_TRAIN_AVG.search(line)) and cur_run is not None:
                run_rates.append([
                    global_counter + cur_episodes,        
                    float(m_avg.group(1)),                 
                    float(m_avg.group(2)),                
                    float(m_avg.group(3)),                
                ])
                continue

            m_eval = RE_EVAL_YN.search(line) or RE_EVAL_NUM.search(line)
            if m_eval and cur_run is not None:
                reward = float(m_eval.group(1))
                raw_succ = m_eval.group(2).strip()
                raw_elim = m_eval.group(3).strip()

                succ_flag = (raw_succ.upper() == "Y") if raw_succ.isalpha() else (float(raw_succ) > 0.0)
                elim_flag = (raw_elim.upper() == "Y") if raw_elim.isalpha() else (float(raw_elim) > 0.0)

                run_results.append([
                    global_counter + cur_episodes,
                    reward, succ_flag, elim_flag
                ])
                continue

            if (m_loss := RE_LOSS.search(line)) and cur_run is not None:
                loss_buf.append(float(m_loss.group(1)))

    # flush final run’s loss buffer
    if cur_run is not None and loss_buf:
        run_losses.append([global_counter + cur_episodes, np.mean(loss_buf)])

    # build DataFrames
    df_runs = pd.DataFrame(
        run_results,
        columns=["CumEpisodes", "Reward", "Success", "Elimination"],
    )
    df_rates = pd.DataFrame(
        run_rates,
        columns=["CumEpisodes", "AvgReward", "SuccessRate", "ElimRate"],
    )
    df_loss = pd.DataFrame(
        run_losses,
        columns=["CumEpisodes", "MeanLoss"],
    )

    # save CSVs
    df_runs.to_csv(RUN_RESULTS_CSV, index=False)
    df_rates.to_csv(RUN_RATES_CSV, index=False)
    df_loss.to_csv(os.path.join(RES_DIR, "run_losses.csv"), index=False)
    print(
        f"Parsed {len(df_runs)} runs → "
        f"{RUN_RESULTS_CSV}, {RUN_RATES_CSV}, run_losses.csv"
    )
    return df_runs, df_rates, df_loss

def graph_bar_chart(df_runs: pd.DataFrame) -> None:
    counts = [
        len(df_runs[~df_runs.Success]),
        len(df_runs[df_runs.Success]),
        len(df_runs[~df_runs.Elimination]),
        len(df_runs[df_runs.Elimination]),
    ]
    labels = ["Run\nFailed", "Run\nPassed", "Elim\nFailed", "Elim\nPassed"]
    colors = ["#1f77b4", "#ff7f0e"] * 2
    xpos   = np.arange(4)

    plt.figure(figsize=(6, 5))
    plt.bar(xpos, counts, color=colors, width=0.6)
    for x, c in zip(xpos, counts):
        plt.text(x, c + 0.5, str(c), ha="center", va="bottom", fontsize=9)

    total = len(df_runs)
    plt.text(3.9, max(counts) + 2, f"Total runs: {total}",
             ha="right", va="bottom", fontsize=9)
    plt.xticks(xpos, labels)
    plt.ylabel("Count")
    plt.title("Run & Elimination Outcomes")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "pass_fail_bar.png"), dpi=300)
    plt.close()

def graph_rate_run(df_stats: pd.DataFrame) -> None:
    df_stats = df_stats.sort_values("CumEpisodes")
    x, succ, elim = df_stats.CumEpisodes, df_stats.SuccessRate, df_stats.ElimRate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # success
    ax1.scatter(x, succ, s=14, color="#ff7f0e", alpha=0.7)
    m1, b1 = np.polyfit(x, succ, 1)
    ax1.plot(x, m1 * x + b1, "--", lw=1.3, color="#ff7f0e")
    ax1.set_ylabel("Success Rate (%)")
    ax1.text(0.02, 0.9, f"Mean = {succ.mean():.1f} %",
             transform=ax1.transAxes, fontsize=8,
             bbox=dict(facecolor="white", edgecolor="black", pad=0.2))

    # elimination
    ax2.scatter(x, elim, s=14, color="#1f77b4", alpha=0.7)
    m2, b2 = np.polyfit(x, elim, 1)
    ax2.plot(x, m2 * x + b2, "--", lw=1.3, color="#1f77b4")
    ax2.set_ylabel("Elim Rate (%)")
    ax2.set_xlabel("Cumulative Episodes")
    ax2.text(0.02, 0.9, f"Mean = {elim.mean():.1f} %",
             transform=ax2.transAxes, fontsize=8,
             bbox=dict(facecolor="white", edgecolor="black", pad=0.2))

    fig.suptitle("Rates over Training")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(RES_DIR, "rate_over_runs.png"), dpi=300)
    plt.close()

def graph_rate_sliding(df_stats: pd.DataFrame, win: int = 50) -> None:
    df_stats = df_stats.sort_values("CumEpisodes")
    runs = df_stats.CumEpisodes.values
    sr   = df_stats.SuccessRate.values
    er   = df_stats.ElimRate.values

    def smooth(arr):
        means, vars_ = [], []
        dq = deque(maxlen=win)
        for x in arr:
            dq.append(x)
            means.append(statistics.mean(dq))
            vars_.append(statistics.pvariance(dq) if len(dq) > 1 else 0)
        return np.array(means), np.array(vars_)

    m_sr, v_sr = smooth(sr)
    m_er, v_er = smooth(er)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(runs, m_sr, color="#ff7f0e", label="Mean Success %")
    ax[0].fill_between(runs, m_sr - np.sqrt(v_sr), m_sr + np.sqrt(v_sr),
                       alpha=0.18, color="#ff7f0e")
    ax[0].set_ylabel("Success %"); ax[0].legend()

    ax[1].plot(runs, m_er, color="#1f77b4", label="Mean Elimination %")
    ax[1].fill_between(runs, m_er - np.sqrt(v_er), m_er + np.sqrt(v_er),
                       alpha=0.18, color="#1f77b4")
    ax[1].set_ylabel("Elimination %")
    ax[1].set_xlabel("Cumulative Episodes"); ax[1].legend()

    fig.suptitle(f"Sliding-window (w={win}) Mean ± SD")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(RES_DIR, "sliding_window_rates.png"), dpi=300)
    plt.close()

def graph_confusion(df_runs: pd.DataFrame) -> None:
    tp = len(df_runs[df_runs.Success & df_runs.Elimination])
    fp = len(df_runs[df_runs.Success & ~df_runs.Elimination])
    fn = len(df_runs[~df_runs.Success & df_runs.Elimination])
    tn = len(df_runs[~df_runs.Success & ~df_runs.Elimination])
    cm = np.array([[tp, fp], [fn, tn]])

    plt.figure(figsize=(4, 3.6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Elim Yes", "Elim No"],
        yticklabels=["Success Yes", "Success No"],
    )
    plt.title("Confusion Matrix of Outcomes"); plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

def graph_reward_run(df_runs: pd.DataFrame) -> None:
    x = df_runs.CumEpisodes.values
    y = df_runs.Reward.values
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(x, y, s=15, color="#4daf4a", alpha=0.8)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, "--", lw=1.8, color="#4daf4a")
    ax.set_title("Evaluation Reward versus Training Progress")
    ax.set_xlabel("Cumulative Episodes"); ax.set_ylabel("Reward")
    ax.text(0.02, 0.9, f"Mean = {y.mean():.2f}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor="white", edgecolor="black", pad=0.2))
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "avg_reward_over_runs.png"), dpi=300)
    plt.close()

def graph_loss(df_loss: pd.DataFrame) -> None:
    if df_loss.empty:
        return
    plt.figure(figsize=(8, 3))
    plt.plot(df_loss.CumEpisodes, df_loss.MeanLoss, marker="o", lw=1.25)
    plt.title("Mean TD-loss per Run")
    plt.xlabel("Cumulative Episodes"); plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "loss_curve.png"), dpi=300)
    plt.close()

def main() -> None:
    df_runs, df_rates, df_loss = parse_data()
    if df_runs.empty:
        print("No runs parsed — nothing to plot."); return

    graph_bar_chart(df_runs)
    graph_rate_run(df_rates)
    graph_rate_sliding(df_rates)
    graph_confusion(df_runs)
    graph_reward_run(df_runs)
    graph_loss(df_loss)

if __name__ == "__main__":
    main()
