import os, re, statistics
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LOG_PATH        = "training.log"
RES_DIR         = "results"
RUN_RESULTS_CSV = os.path.join(RES_DIR, "run_results.csv")
RUN_RATES_CSV   = os.path.join(RES_DIR, "run_rates.csv")

os.makedirs(RES_DIR, exist_ok=True)

def parse_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, stats = [], []
    run_id = 0

    with open(LOG_PATH, "rb") as f:
        for raw in f:
            line = raw.decode("latin1", errors="ignore")

            m_eval = re.search(
                r"Evaluation Reward:\s*([-\d.]+)\s*\|\s*(Passed|Failed)\s*\|\s*"
                r"(Eliminated|Did not eliminate)", line)
            if m_eval:
                run_id += 1
                reward = float(m_eval.group(1))
                passed = m_eval.group(2) == "Passed"
                elim   = m_eval.group(3).startswith("Eliminated")
                runs.append([run_id, reward, passed, elim])

            m_avg = re.search(
                r"Avg Reward:\s*([-\d.]+)\s*\|\s*Success Rate:\s*([\d.]+)%\s*\|\s*"
                r"Elimination Rate:\s*([\d.]+)%", line)
            if m_avg:
                stats.append([run_id,
                              float(m_avg.group(1)),
                              float(m_avg.group(2)),
                              float(m_avg.group(3))])

    df_runs  = pd.DataFrame(runs,  columns=["Run", "Reward", "Success", "Elimination"])
    df_stats = (pd.DataFrame(stats,
                             columns=["Run", "AvgReward", "SuccessRate", "ElimRate"])
                .groupby("Run").mean().reset_index())

    df_runs.to_csv(RUN_RESULTS_CSV, index=False)
    df_stats.to_csv(RUN_RATES_CSV,  index=False)
    print(f"Parsed {len(df_runs)} runs → {RUN_RESULTS_CSV}, {RUN_RATES_CSV}")
    return df_runs, df_stats


def graph_bar_chart(df_runs: pd.DataFrame):
    counts = [
        len(df_runs[~df_runs.Success]),
        len(df_runs[df_runs.Success]),
        len(df_runs[~df_runs.Elimination]),
        len(df_runs[df_runs.Elimination]),
    ]
    labels  = ["Run\nFailed", "Run\nPassed", "Elim\nFailed", "Elim\nPassed"]
    colors  = ["#1f77b4", "#ff7f0e"] * 2
    xpos    = np.arange(4)

    plt.figure(figsize=(6, 5))
    plt.bar(xpos, counts, color=colors, width=0.6)
    for x, c in zip(xpos, counts):
        plt.text(x, c + 0.5, str(c), ha="center", va="bottom", fontsize=9)

    total = len(df_runs)
    plt.text(3.95, max(counts) + 2, f"Total runs: {total}",
             ha="right", va="bottom", fontsize=9)
    plt.xticks(xpos, labels)
    plt.ylabel("Count")
    plt.title("Run & Elimination Outcomes")
    plt.tight_layout()

    out = os.path.join(RES_DIR, "pass_fail_bar.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Bar chart {out}")


def graph_rate_run(df_stats: pd.DataFrame):
    """Per-run scatter plot (success & elimination) + best-fit lines."""
    df_stats = df_stats.sort_values("Run")
    x = df_stats.Run.values
    succ = df_stats.SuccessRate.values
    elim = df_stats.ElimRate.values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.scatter(x, succ, s=12, color="#ff7f0e", alpha=0.7)
    m1, b1 = np.polyfit(x, succ, 1)
    ax1.plot(x, m1 * x + b1, "--", linewidth=1.4, color="#ff7f0e")
    ax1.set_ylabel("Success Rate (%)")
    ax1.text(0.02, 0.9, f"Mean = {succ.mean():.1f}%",
             transform=ax1.transAxes, fontsize=8,
             bbox=dict(facecolor="white", edgecolor="black", pad=0.2))

   
    ax2.scatter(x, elim, s=12, color="#1f77b4", alpha=0.7)
    m2, b2 = np.polyfit(x, elim, 1)
    ax2.plot(x, m2 * x + b2, "--", linewidth=1.4, color="#1f77b4")
    ax2.set_ylabel("Elim Rate (%)")
    ax2.set_xlabel("Run")
    ax2.text(0.02, 0.9, f"Mean = {elim.mean():.1f}%",
             transform=ax2.transAxes, fontsize=8,
             bbox=dict(facecolor="white", edgecolor="black", pad=0.2))

    fig.suptitle("Rates over Runs")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(RES_DIR, "rate_over_runs.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Rate-over-runs plot {out}")


def graph_rate_sliding(df_stats: pd.DataFrame, win: int = 50):
    df_stats = df_stats.sort_values("Run")
    runs = df_stats.Run.values
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
                       alpha=0.2, color="#ff7f0e")
    ax[0].set_ylabel("Success %")
    ax[0].legend()

    ax[1].plot(runs, m_er, color="#1f77b4", label="Mean Elimination %")
    ax[1].fill_between(runs, m_er - np.sqrt(v_er), m_er + np.sqrt(v_er),
                       alpha=0.2, color="#1f77b4")
    ax[1].set_ylabel("Elimination %")
    ax[1].set_xlabel("Run")
    ax[1].legend()

    fig.suptitle(f"Sliding-window (w={win}) Mean ± SD", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(RES_DIR, "sliding_window_rates.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Sliding-window plot {out}")


def graph_confusion(df_runs: pd.DataFrame):
    tp = len(df_runs[df_runs.Success & df_runs.Elimination])
    fp = len(df_runs[df_runs.Success & ~df_runs.Elimination])
    fn = len(df_runs[~df_runs.Success & df_runs.Elimination])
    tn = len(df_runs[~df_runs.Success & ~df_runs.Elimination])
    cm = np.array([[tp, fp], [fn, tn]])

    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Elim Yes", "Elim No"],
                yticklabels=["Success Yes", "Success No"])
    plt.title("Confusion Matrix of Outcomes")
    plt.tight_layout()

    out = os.path.join(RES_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Confusion matrix{out}")


def graph_reward_run(df_runs: pd.DataFrame):
    x = df_runs.Run.values
    y = df_runs.Reward.values

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(x, y, s=15, color="#4daf4a", alpha=0.8)

    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, "--", linewidth=1.8, color="#4daf4a")

    ax.set_title("Average Reward over Runs")
    ax.set_xlabel("Run")
    ax.set_ylabel("Evaluation Reward")
    ax.text(0.02, 0.9, f"Mean = {y.mean():.2f}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor="white", edgecolor="black", pad=0.2))

    fig.tight_layout()

    out = os.path.join(RES_DIR, "avg_reward_over_runs.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Reward plot{out}")

def main():
    df_runs, df_stats = parse_data()
    graph_bar_chart(df_runs)
    graph_rate_run(df_stats)       
    graph_rate_sliding(df_stats)   
    graph_confusion(df_runs)
    graph_reward_run(df_runs)


if __name__ == "__main__":
    main()
