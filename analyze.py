"""Aggregates the runs of exp.py: figures and statistical tests."""

import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RES = Path("results")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)

COLORS = {"ddpg": "#2a78d6", "td3": "#eb6834"}
STYLES = {0: "-", 1: "--"}
LABEL = {"ddpg": "DDPG", "td3": "TD3"}
BIAS_FROM = 50_000
FINAL_EVALS = 3

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight",
})


def load():
    runs = {}
    for f in sorted(RES.glob("*.json")):
        r = json.loads(f.read_text())
        c = r["config"]
        key = (c["algo"], int(c["layer_norm"]), c["depth"])
        runs.setdefault(key, []).append(r)
    return runs


def series(run, field):
    h = run["history"]
    return np.array([e["step"] for e in h]), np.array([e[field] for e in h])


def boot_ci(x, n=10_000):
    x = np.asarray(x)
    means = RNG.choice(x, (n, len(x))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def run_metrics(run):
    steps, ret = series(run, "return_mean")
    _, nb = series(run, "nbias_q1")
    _, b = series(run, "bias_q1")
    late = steps >= BIAS_FROM
    return {
        "final": ret[-FINAL_EVALS:].mean(),
        "auc": ret[steps >= 10_000].mean(),
        "nbias": nb[late].mean(),
        "bias": b[late].mean(),
        "final_nbias": nb[-FINAL_EVALS:].mean(),
    }


def curves(runs, depths):
    fields = [("return_mean", "Evaluation return"),
              ("bias_q1", "Bias  Q − G  (symlog)")]
    fig, axes = plt.subplots(2, len(depths), figsize=(3.4 * len(depths), 4.6),
                             sharex=True, squeeze=False)
    for j, d in enumerate(depths):
        for i, (field, ylabel) in enumerate(fields):
            ax = axes[i, j]
            for algo, ln in product(["ddpg", "td3"], [0, 1]):
                rs = runs.get((algo, ln, d), [])
                if not rs:
                    continue
                n = min(len(series(r, field)[0]) for r in rs)
                steps = series(rs[0], field)[0][:n]
                ys = np.stack([series(r, field)[1][:n] for r in rs])
                m = ys.mean(0)
                ci = np.array([boot_ci(ys[:, k], 2000) for k in range(n)])
                ax.plot(steps / 1e3, m, STYLES[ln], color=COLORS[algo], lw=1.6,
                        label=f"{LABEL[algo]}{' + LN' if ln else ''} (n={len(rs)})")
                ax.fill_between(steps / 1e3, ci[:, 0], ci[:, 1],
                                color=COLORS[algo], alpha=0.12, lw=0)
            if field == "bias_q1":
                ax.axhline(0, color="#52514e", lw=0.8)
                ax.set_yscale("symlog", linthresh=10)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(f"{d} hidden layers")
            if i == 1:
                ax.set_xlabel("Environment steps (×1000)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 1.04))
    fig.savefig(FIG / "curves.pdf"); fig.savefig(FIG / "curves.svg")
    fig.savefig(FIG / "curves.png", dpi=150)


def scatter(runs):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    xs, ys = [], []
    for (algo, ln, d), rs in sorted(runs.items()):
        for r in rs:
            m = run_metrics(r)
            xs.append(m["bias"]); ys.append(m["final"])
            ax.scatter(m["bias"], m["final"], s=22, color=COLORS[algo],
                       marker="o" if ln == 0 else "^",
                       facecolors="none" if d == 3 else COLORS[algo], lw=1.2)
    rho, p = stats.spearmanr(xs, ys)
    ax.set_xlabel("Mean bias Q − G (≥ 50k steps, symlog)")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_ylabel("Final return")
    ax.set_title(f"All runs: Spearman ρ = {rho:.2f} (p = {p:.1g})", fontsize=8.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLORS["ddpg"], marker="o", ls="", label="DDPG"),
         Line2D([], [], color=COLORS["td3"], marker="o", ls="", label="TD3"),
         Line2D([], [], color="#52514e", marker="o", ls="", label="no LN"),
         Line2D([], [], color="#52514e", marker="^", ls="", label="LN"),
         Line2D([], [], color="#52514e", marker="o", mfc="none", ls="", label="3 layers")]
    ax.legend(handles=h, fontsize=7, loc="best")
    fig.savefig(FIG / "scatter.pdf"); fig.savefig(FIG / "scatter.svg")
    fig.savefig(FIG / "scatter.png", dpi=150)
    return rho, p


def welch(a, b):
    t = stats.ttest_ind(a, b, equal_var=False)
    return t.statistic, t.pvalue


def main():
    runs = load()
    depths = sorted({k[2] for k in runs})
    curves(runs, depths)
    rho, p = scatter(runs)

    lines = ["| algo | LN | depth | n | final return [95% CI] | AUC | mean bias [95% CI] | mean nbias |",
             "|---|---|---|---|---|---|---|---|"]
    M = {}
    for key in sorted(runs):
        ms = [run_metrics(r) for r in runs[key]]
        M[key] = {k: np.array([m[k] for m in ms]) for k in ms[0]}
        f, nb = M[key]["final"], M[key]["bias"]
        cf, cb = boot_ci(f), boot_ci(nb)
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(f)} | {f.mean():.1f} [{cf[0]:.1f}, {cf[1]:.1f}] "
            f"| {M[key]['auc'].mean():.1f} | {nb.mean():.1f} [{cb[0]:.1f}, {cb[1]:.1f}] "
            f"| {M[key]['nbias'].mean():.2f} |")

    lines += ["", "Welch t-tests (two-sided), per depth:", "",
              "| comparison | depth | metric | t | p | p (Holm) |", "|---|---|---|---|---|---|"]
    tests = []
    for d in depths:
        for algo in ["ddpg", "td3"]:
            a, b = M.get((algo, 0, d)), M.get((algo, 1, d))
            if a is None or b is None:
                continue
            for metric in ["bias", "final", "auc"]:
                t, pv = welch(a[metric], b[metric])
                tests.append((f"{algo}: no LN vs LN", d, metric, t, pv))
        for ln in [0, 1]:
            a, b = M.get(("ddpg", ln, d)), M.get(("td3", ln, d))
            if a is None or b is None:
                continue
            for metric in ["bias", "final", "auc"]:
                t, pv = welch(a[metric], b[metric])
                tests.append((f"LN={ln}: DDPG vs TD3", d, metric, t, pv))
    pvals = np.array([x[4] for x in tests])
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    for (name, d, metric, t, pv), pa in zip(tests, adj):
        lines.append(f"| {name} | {d} | {metric} | {t:.2f} | {pv:.3g} | {pa:.3g} |")

    lines += ["", f"Spearman (mean bias vs final return, all runs): rho={rho:.3f}, p={p:.3g}"]
    for algo in ["ddpg", "td3"]:
        x = np.concatenate([M[k]["bias"] for k in M if k[0] == algo])
        y = np.concatenate([M[k]["final"] for k in M if k[0] == algo])
        r, pv = stats.spearmanr(x, y)
        lines.append(f"Spearman within {algo}: rho={r:.3f}, p={pv:.3g} (n={len(x)})")
    out = "\n".join(lines)
    Path("summary.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
