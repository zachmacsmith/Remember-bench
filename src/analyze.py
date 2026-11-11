"""Pilot analysis.

Sections (report.md) and charts (figures/*.png):
  1 data quality            fig1 base-rate heatmap
  2 AUROC matrix            fig2 predictor x target heatmap
  3 self-advantage (H1)     fig3 delta forest plot (paired bootstrap CIs)
  4 within-text AUROC       fig4 pooled vs within-text dot plot
  5 incremental information (logistic likelihood-ratio test: self over text + frequency + siblings)
  6 outsider baselines (tier, n-gram frequency, other models' actual recall)
  7 calibration             fig5 reliability curves
  8 framing, prompt-variant sensitivity, largest disagreements, scoring checks, kill criteria

  python -I src/analyze.py [--config pilot_claude] [--outcome gist|verbatim]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, read_jsonl, load_config  # noqa: E402

B = 2000
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3 (validated all-pairs)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ---------------------------------------------------------------- metrics
def auroc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, float)
    ok = ~(np.isnan(s) | np.isnan(y))
    s, y = s[ok], y[ok].astype(int)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def within_auroc(score, y, strata):
    """n-weighted mean of AUROC computed inside each stratum that has both outcomes."""
    num = den = 0.0
    for g in pd.unique(strata):
        m = strata == g
        a = auroc(score[m], y[m])
        if not np.isnan(a):
            w = int((~np.isnan(np.asarray(y[m], float))).sum())
            num += a * w
            den += w
    return num / den if den else np.nan


def boot_diff(f_a, f_b, n, strata=None, rng=None):
    """Paired bootstrap of f_a(ix) - f_b(ix); stratified resampling if strata given."""
    d = []
    groups = None if strata is None else [np.where(strata == g)[0] for g in pd.unique(strata)]
    for _ in range(B):
        if groups is None:
            ix = rng.integers(0, n, n)
        else:
            ix = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        a, b = f_a(ix), f_b(ix)
        if not (np.isnan(a) or np.isnan(b)):
            d.append(a - b)
    if len(d) < B / 2:
        return np.nan, np.nan, np.nan
    d = np.array(d)
    return np.percentile(d, 2.5), np.percentile(d, 97.5), float((d <= 0).mean())


def logit_fit(X, y, ridge=1e-2, iters=50):
    """Newton-Raphson logistic regression with a small ridge (handles near-separation). Returns loglik, beta."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        g = X.T @ (y - p) - ridge * beta
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(len(beta))
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    p = np.clip(1 / (1 + np.exp(-(X @ beta))), 1e-12, 1 - 1e-12)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum()), beta


def chi2_sf_df1(x):
    return math.erfc(math.sqrt(max(x, 0) / 2))


def calib(prob, y, bins=5):
    p, y = np.asarray(prob, float) / 100, np.asarray(y, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    p, y = p[ok], y[ok]
    if not len(p):
        return {}
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece, curve = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.sum():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            curve.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return dict(mean_stated=p.mean(), observed=y.mean(), overconf=p.mean() - y.mean(),
                brier=((p - y) ** 2).mean(), ece=ece, curve=curve)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- charts
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    })
    return plt


def heatmap(plt, df, path, title, vmin, vmax, fmt_s="{:.2f}", note=None, outline_diag=False):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(1.6 + 1.25 * df.shape[1], 0.9 + 0.45 * df.shape[0]))
    v = df.to_numpy(float)
    ax.imshow(v, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=0)
    ax.set_yticks(range(df.shape[0]), df.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isnan(v[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", color=INK2, fontsize=9)
                continue
            frac = (v[i, j] - vmin) / (vmax - vmin)
            ax.text(j, i, fmt_s.format(v[i, j]), ha="center", va="center", fontsize=9,
                    color="#ffffff" if frac > 0.55 else INK)
            if outline_diag and i == j:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec=INK, lw=1.5))
    # 2px surface gaps between cells
    for k in range(1, v.shape[1]):
        ax.axvline(k - 0.5, color=SURF, lw=2)
    for k in range(1, v.shape[0]):
        ax.axhline(k - 0.5, color=SURF, lw=2)
    ax.set_title(title, loc="left")
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def forest(plt, rows, path, title):
    """rows: list of (label, est, lo, hi, group_index)."""
    fig, ax = plt.subplots(figsize=(7.5, 0.5 + 0.34 * len(rows)))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, est, lo, hi, g) in zip(ys, rows):
        if np.isnan(est):
            continue
        ax.plot([lo, hi], [y, y], color=SERIES[g % 3], lw=2, solid_capstyle="round")
        ax.plot(est, y, "o", ms=8, color=SERIES[g % 3], mec=SURF, mew=2)
        ax.text(hi + 0.01 if not np.isnan(hi) else est + 0.01, y, f"{est:+.3f}", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Δ AUROC (self C1 minus comparator), 95% paired bootstrap CI")
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def dots(plt, data, path, title):
    """data: {target: {method: (pooled, within)}} -> small multiples, one panel per target."""
    targets = list(data)
    nmax = max(len(v) for v in data.values())
    fig, axes = plt.subplots(1, len(targets), figsize=(3.6 * len(targets), 0.9 + 0.42 * nmax))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        methods = list(data[t])
        ys = np.arange(len(methods))[::-1]
        for y, m in zip(ys, methods):
            pooled, within = data[t][m]
            if not np.isnan(pooled) and not np.isnan(within):
                ax.plot([pooled, within], [y, y], color=GRID, lw=2)
            ax.plot(pooled, y, "o", ms=8, color=SERIES[0], mec=SURF, mew=2)
            ax.plot(within, y, "D", ms=7, color=SERIES[1], mec=SURF, mew=2)
        ax.axvline(0.5, color=INK2, lw=1)
        ax.set_xlim(0.3, 1.0)
        ax.set_title(f"target: {t}", loc="left", fontsize=10)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(ys, methods)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=SERIES[0], ms=8, label="pooled AUROC"),
                        Line2D([], [], marker="D", ls="", color=SERIES[1], ms=7, label="within-text AUROC")],
               loc="upper right", frameon=False, ncol=2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def reliability(plt, curves, path, title):
    """curves: {model: [(stated, observed, n), ...]}"""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.62, 0.55, "perfect calibration", rotation=0, ha="left", fontsize=8, color=INK2)
    for k, (m, c) in enumerate(curves.items()):
        if not c:
            continue
        c = [b for b in c if b[2] >= 5]
        if not c:
            continue
        xs, ys, ns = zip(*c)
        ax.plot(xs, ys, color=SERIES[k], lw=2)
        ax.scatter(xs, ys, s=[max(30, 4 * n) for n in ns], color=SERIES[k], edgecolor=SURF, linewidth=2, zorder=3, label=m)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("stated P(gist-correct), binned")
    ax.set_ylabel("observed gist-correct rate")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(title, loc="left")
    fig.text(0.01, 0.01, "Below the dashed line = overconfident. Marker area ∝ items in bin; bins with < 5 items hidden.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summary_chart(plt, data, path):
    """data: {target: {"pooled": (self, best_other, best_other_name), "within": (...)}} -> grouped bars from the chance line."""
    targets = list(data)
    fig, axes = plt.subplots(1, len(targets), figsize=(3.4 * len(targets), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.36
    for ax, t in zip(axes, targets):
        for gi, key in enumerate(("pooled", "within")):
            s_, o_, oname = data[t][key]
            for k, (val, colr) in enumerate(((s_, SERIES[0]), (o_, SERIES[1]))):
                x = gi + (k - 0.5) * w
                if np.isnan(val):
                    continue
                ax.bar(x, val - 0.5, w, bottom=0.5, color=colr, edgecolor=SURF, linewidth=2)
                above = val >= 0.5
                ax.text(x, val + (0.012 if above else -0.012), f"{val:.2f}", ha="center", va="bottom" if above else "top", fontsize=9, color=INK)
        ax.axhline(0.5, color=INK2, lw=1)
        ax.set_xticks([0, 1], [f"all items\n(other: {data[t]['pooled'][2]})", f"within each text\n(other: {data[t]['within'][2]})"], fontsize=9)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0.4, 1.02)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"Predicting {t}'s recall", loc="left", fontsize=10)
    axes[0].set_ylabel("AUROC (0.5 = chance)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=SERIES[0], label="the model itself (C1)"), Patch(color=SERIES[1], label="best other model (C3)")],
               loc="upper right", frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Does a model predict its own recall better than other models can?", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, 0.01, "Bars start at chance (0.5). 'Within each text' compares items only against others from the same text, "
             "removing the easy-text/hard-text signal.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot")
    ap.add_argument("--outcome", default="gist", choices=["gist", "verbatim"])
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--exclude-refusals", action="store_true", help="treat refused attempts as missing instead of grade 0")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / cfg["out_dir"]
    fig_dir = out_dir / "figures"
    rng = np.random.default_rng(0)
    items = pd.DataFrame(read_jsonl(ROOT / cfg["items"])).set_index("item_id")
    pop_path = ROOT / "data" / "items" / "pilot_popularity.csv"
    if pop_path.exists():
        items = items.join(pd.read_csv(pop_path).set_index("item_id")[["log_freq"]])
    pred = pd.read_csv(out_dir / "predictions.csv")
    rec = pd.read_csv(out_dir / "recall.csv")
    models = [m["id"] for m in cfg["models"]]
    pred = pred[pred.predictor.isin(models) & pred.target.isin(models)]
    rec = rec[rec.model_id.isin(models)]
    short = {m["id"]: m["id"].replace("claude-", "") for m in cfg["models"]}
    match = {m["id"]: m["match"] for m in cfg["models"]}
    thr = 2 if args.outcome == "gist" else 3
    col = args.outcome
    ids = list(items.index)
    text = items["text_id"].reindex(ids).to_numpy()
    strata_lbl = (items["text_id"] + "/" + items["popularity_tier"]).reindex(ids)

    if args.exclude_refusals:
        rec.loc[rec.refusal.astype(bool), "grade"] = np.nan
        out_dir = out_dir / "excl_refusals"
        out_dir.mkdir(exist_ok=True)
        fig_dir = out_dir / "figures"
    rec["y"] = np.where(rec["grade"].isna(), np.nan, (rec["grade"] >= thr).astype(float))
    Y = rec.pivot_table(index="item_id", columns="model_id", values="y").reindex(ids)
    G = rec.pivot_table(index="item_id", columns="model_id", values="grade").reindex(ids)

    def score(p, t, cond, field=col):
        sub = pred[(pred.predictor == p) & (pred.target == t) & (pred.condition == cond)]
        return sub.set_index("item_id")[field].reindex(ids).astype(float).to_numpy()

    def S(p, t):  # the primary prediction of p about t
        return score(p, t, "C1" if p == t else "C3")

    L, summ = [], {"outcome": args.outcome, "threshold": thr, "n_items": len(ids), "models": {}}
    add = L.append
    add(f"# Pilot report: {cfg['name']} (outcome: {args.outcome}, grade ≥ {thr})\n")
    add(f"{len(ids)} items · models: {', '.join(short[m] for m in models)} · judge: {cfg['judge']['id']} · backend: {cfg.get('backend', 'openrouter')}\n")
    n_graded = {m: int(Y[m].notna().sum()) for m in models}
    add("Graded recall attempts per model: " + ", ".join(f"{short[m]} {n_graded[m]}" for m in models) + "\n")

    # ---- 1 data quality
    add("## 1. Data quality\n")
    add("| model | base rate | grades 0/1/2/3 | refusals | C1 parse fail | C3 parse fail | C1 leaks | C1 thinking tok (max) | C1 gist SD | C1 distinct |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for m in models:
        r = rec[rec.model_id == m]
        c1 = pred[(pred.predictor == m) & (pred.condition == "C1")]
        c3 = pred[(pred.predictor == m) & (pred.condition == "C3")]
        dist = [int((r.grade == g).sum()) for g in range(4)]
        s = dict(base_rate=float(np.nanmean(Y[m])), grade_dist=dist, refusals=int(r.refusal.sum()),
                 c1_parse_fail=float(1 - c1.parse_ok.mean()), c3_parse_fail=float(1 - c3.parse_ok.mean()),
                 c1_leaks=int(c1.get("leak", pd.Series(dtype=bool)).sum()),
                 c1_think_max=float(c1.get("thinking_tokens", pd.Series([0])).fillna(0).max()),
                 c1_gist_sd=float(c1[col].std()), c1_distinct=int(c1[col].nunique()))
        summ["models"][m] = s
        add(f"| {short[m]} | {fmt(s['base_rate'], 2)} | {'/'.join(map(str, dist))} | {s['refusals']} | {fmt(s['c1_parse_fail'], 2)} | "
            f"{fmt(s['c3_parse_fail'], 2)} | {s['c1_leaks']} | {s['c1_think_max']:.0f} | {fmt(s['c1_gist_sd'], 1)} | {s['c1_distinct']} |")
    add("")
    br = rec.assign(stratum=rec.item_id.map(strata_lbl)).pivot_table(index="stratum", columns="model_id", values="y", aggfunc="mean")
    br = br[models].rename(columns=short)
    add("Recall base rate by text/tier:\n")
    add(br.round(2).to_markdown())
    add("")
    ref = rec[rec.refusal].assign(text=rec.item_id.map(items["text_id"])).groupby(["text", "model_id"]).size()
    if len(ref):
        add("Refusals by text: " + "; ".join(f"{t}/{short[m]}: {n}" for (t, m), n in ref.items()) + "\n")

    # ---- 2 AUROC matrices
    add("## 2. AUROC matrix (rows predict columns; diagonal = self C1, off-diagonal = C3)\n")
    mats = {}
    for oc, f in (("gist", "gist"), ("verbatim", "verbatim")):
        yy = rec.assign(v=np.where(rec.grade.isna(), np.nan, (rec.grade >= (2 if oc == "gist" else 3)).astype(float))) \
                .pivot_table(index="item_id", columns="model_id", values="v").reindex(ids)
        mat = pd.DataFrame(index=[short[m] for m in models], columns=[short[m] for m in models], dtype=float)
        for p in models:
            for t in models:
                mat.loc[short[p], short[t]] = auroc(score(p, t, "C1" if p == t else "C3", f), yy[t].to_numpy())
        mats[oc] = mat
        add(f"**{oc}** (outcome grade ≥ {2 if oc == 'gist' else 3}, predictor field `{f}`)\n")
        add(mat.round(3).to_markdown())
        add("")
    summ["auroc_matrix"] = {k: v.round(4).to_dict() for k, v in mats.items()}

    # ---- 3 self-advantage
    add("## 3. Self-advantage (H1)\n")
    add("Δ = AUROC(self C1 → M) − AUROC(comparator → M), same items, paired bootstrap (2,000 resamples). "
        "Comparators: each sibling's C3 prediction, and M's own prediction under the C3 'another model' framing.\n")
    add("| target | comparator | self AUROC | comparator AUROC | Δ | 95% CI | P(Δ ≤ 0) |")
    add("|---|---|---|---|---|---|---|")
    forest_rows = []
    for gi, t in enumerate(models):
        y = Y[t].to_numpy()
        s_self = S(t, t)
        comps = [(f"{short[p]} (C3)" + (" [matched]" if p == match[t] else ""), score(p, t, "C3")) for p in models if p != t]
        comps.append(("self, C3 framing", score(t, t, "C3self")))
        summ["models"][t]["self_auroc"] = auroc(s_self, y)
        summ["models"][t]["comparisons"] = {}
        for lab, s_c in comps:
            a_s, a_c = auroc(s_self, y), auroc(s_c, y)
            lo, hi, p0 = boot_diff(lambda ix: auroc(s_self[ix], y[ix]), lambda ix: auroc(s_c[ix], y[ix]), len(y), rng=rng)
            summ["models"][t]["comparisons"][lab] = dict(comp_auroc=a_c, delta=a_s - a_c, ci=[lo, hi], p_le_0=p0)
            add(f"| {short[t]} | {lab} | {fmt(a_s)} | {fmt(a_c)} | {fmt(a_s - a_c)} | [{fmt(lo)}, {fmt(hi)}] | {fmt(p0, 3)} |")
            forest_rows.append((f"{short[t]}  vs {lab}", a_s - a_c, lo, hi, gi))
    add("")

    # ---- 4 within-text
    add("## 4. Within-text AUROC (removes between-text popularity)\n")
    add("AUROC computed inside each text (n-weighted mean over texts where the target has both outcomes). "
        "A predictor that only knows 'KJV is easy, Tennessee is hard' scores ~0.5 here.\n")
    add("| target | predictor | pooled AUROC | within-text AUROC | Δ within (self − predictor) | 95% CI (stratified bootstrap) |")
    add("|---|---|---|---|---|---|")
    dot_data = {}
    for t in models:
        y = Y[t].to_numpy()
        s_self = S(t, t)
        dot_data[short[t]] = {}
        preds = [(f"self ({short[t]})", s_self)] + [(f"{short[p]} (C3)", score(p, t, "C3")) for p in models if p != t]
        preds += [("log n-gram freq", items["log_freq"].reindex(ids).to_numpy() if "log_freq" in items else np.full(len(ids), np.nan))]
        for lab, s in preds:
            pooled, within = auroc(s, y), within_auroc(s, y, text)
            dot_data[short[t]][lab.replace(f" ({short[t]})", "")] = (pooled, within)
            if lab.startswith("self"):
                add(f"| {short[t]} | {lab} | {fmt(pooled)} | {fmt(within)} | | |")
                summ["models"][t]["self_within_auroc"] = within
                continue
            lo, hi, _ = boot_diff(lambda ix: within_auroc(s_self[ix], y[ix], text[ix]), lambda ix: within_auroc(s[ix], y[ix], text[ix]),
                                  len(y), strata=text, rng=rng)
            add(f"| {short[t]} | {lab} | {fmt(pooled)} | {fmt(within)} | {fmt(within_auroc(s_self, y, text) - within)} | [{fmt(lo)}, {fmt(hi)}] |")
    add("")

    # ---- 5 incremental information
    add("## 5. Incremental information: does self-confidence predict M's recall beyond what outsiders know?\n")
    add("Logistic regression of M's outcome on text fixed effects + log n-gram frequency + both siblings' C3 predictions (base), "
        "then adding M's own C1 prediction (full). Likelihood-ratio test, 1 df. Reverse test: does each sibling add beyond self + controls? "
        "Predictors standardized; small ridge (λ=0.01) for stability.\n")
    add("| target | n | LR χ² (self added) | p | self coef | LR χ² (best sibling added to self+controls) | p |")
    add("|---|---|---|---|---|---|---|")
    for t in models:
        y = Y[t].to_numpy()
        others = [p for p in models if p != t]
        cols = {"self": S(t, t), **{p: score(p, t, "C3") for p in others}}
        lf = items["log_freq"].reindex(ids).to_numpy() if "log_freq" in items else np.zeros(len(ids))
        ok = ~np.isnan(y) & ~np.isnan(lf)
        for v in cols.values():
            ok &= ~np.isnan(v)
        if ok.sum() < 30 or y[ok].min() == y[ok].max():
            add(f"| {short[t]} | {int(ok.sum())} | n/a | | | | |")
            continue
        z = lambda v: (v[ok] - v[ok].mean()) / (v[ok].std() + 1e-9)
        fe = pd.get_dummies(pd.Series(text[ok]), drop_first=True).to_numpy(float)
        base_X = np.column_stack([np.ones(ok.sum()), fe, z(lf)] + [z(cols[p]) for p in others])
        ll0, _ = logit_fit(base_X, y[ok])
        ll1, b1 = logit_fit(np.column_stack([base_X, z(cols["self"])]), y[ok])
        lr = 2 * (ll1 - ll0)
        rev = []
        for p in others:
            Xs = np.column_stack([np.ones(ok.sum()), fe, z(lf), z(cols["self"])] + [z(cols[q]) for q in others if q != p])
            llr0, _ = logit_fit(Xs, y[ok])
            llr1, _ = logit_fit(np.column_stack([Xs, z(cols[p])]), y[ok])
            rev.append((2 * (llr1 - llr0), p))
        best_rev = max(rev)
        summ["models"][t]["lr_self"] = dict(chi2=lr, p=chi2_sf_df1(lr), coef=float(b1[-1]))
        add(f"| {short[t]} | {int(ok.sum())} | {fmt(lr, 2)} | {fmt(chi2_sf_df1(lr), 4)} | {fmt(float(b1[-1]), 2)} | "
            f"{fmt(best_rev[0], 2)} ({short[best_rev[1]]}) | {fmt(chi2_sf_df1(best_rev[0]), 4)} |")
    add("")

    # ---- 6 outsider baselines
    add("## 6. Outsider baselines (not in the spec)\n")
    add("| target | self C1 | popularity tier | log n-gram freq | siblings' mean actual grade | best sibling's actual grade |")
    add("|---|---|---|---|---|---|")
    tier = items["popularity_tier"].map({"famous": 1, "shallow": 1, "random": 0, "deep": 0, "obscure": 0}).reindex(ids).astype(float).to_numpy()
    for t in models:
        y = Y[t].to_numpy()
        others = [p for p in models if p != t]
        a_lf = auroc(items["log_freq"].reindex(ids).to_numpy(), y) if "log_freq" in items else np.nan
        a_cons = auroc(G[others].mean(axis=1).to_numpy(), y)
        a_best = max(auroc(G[p].to_numpy(), y) for p in others)
        summ["models"][t].update(tier_auroc=auroc(tier, y), logfreq_auroc=a_lf, consensus_auroc=a_cons, best_sibling_actual_auroc=a_best)
        add(f"| {short[t]} | {fmt(auroc(S(t, t), y))} | {fmt(auroc(tier, y))} | {fmt(a_lf)} | {fmt(a_cons)} | {fmt(a_best)} |")
    add("")

    # ---- 7 calibration
    add("## 7. Calibration of stated P(gist)\n")
    add("| target | predictor | mean stated | observed | overconfidence | Brier | ECE (5 bins) |")
    add("|---|---|---|---|---|---|---|")
    curves = {}
    for t in models:
        y = Y[t].to_numpy()
        for p in models:
            c = calib(S(p, t) if col == "gist" else score(p, t, "C1" if p == t else "C3", "gist"), y)
            if not c:
                continue
            if p == t:
                curves[short[t]] = c["curve"]
                summ["models"][t]["calibration_self"] = {k: v for k, v in c.items() if k != "curve"}
            add(f"| {short[t]} | {short[p]}{' (self)' if p == t else ''} | {fmt(c['mean_stated'], 2)} | {fmt(c['observed'], 2)} | "
                f"{c['overconf']:+.2f} | {fmt(c['brier'], 3)} | {fmt(c['ece'], 3)} |")
    add("")

    # ---- 8 framing, variants, disagreements, scoring
    add("## 8. Framing control, prompt sensitivity, disagreements, scoring checks\n")
    add("**Framing control** (same model, same items: C1 'you' vs C3 'another AI model, {own name}')\n")
    add("| model | AUROC C1 | AUROC C3-self | mean stated gist C1 | mean stated gist C3-self |")
    add("|---|---|---|---|---|")
    for m in models:
        y = Y[m].to_numpy()
        a, b = score(m, m, "C1"), score(m, m, "C3self")
        add(f"| {short[m]} | {fmt(auroc(a, y))} | {fmt(auroc(b, y))} | {fmt(np.nanmean(a), 1)} | {fmt(np.nanmean(b), 1)} |")
    add("")
    add("**Prompt-variant sensitivity** (self C1 AUROC and base rate by template paraphrase)\n")
    var = items["prompt_variant"].reindex(ids).to_numpy()
    add("| model | AUROC variant a | AUROC variant b | base rate a | base rate b | n a / n b |")
    add("|---|---|---|---|---|---|")
    for m in models:
        y, s = Y[m].to_numpy(), S(m, m)
        ra, rb = var == "a", var == "b"
        add(f"| {short[m]} | {fmt(auroc(s[ra], y[ra]))} | {fmt(auroc(s[rb], y[rb]))} | {fmt(np.nanmean(y[ra]), 2)} | {fmt(np.nanmean(y[rb]), 2)} | {ra.sum()} / {rb.sum()} |")
    add("")
    add("**Largest self-vs-matched disagreements** (stated gist; outcome = target's actual grade)\n")
    add("| target | item | locator | self | matched | grade | who was closer |")
    add("|---|---|---|---|---|---|---|")
    for t in models:
        s_self, s_m = S(t, t), score(match[t], t, "C3")
        d = pd.DataFrame({"self": s_self, "other": s_m, "grade": G[t].to_numpy(), "y": Y[t].to_numpy()}, index=ids)
        d["gap"] = (d.self - d.other).abs()
        for iid, r in d.dropna().sort_values("gap", ascending=False).head(4).iterrows():
            closer = "self" if abs(r.self / 100 - r.y) < abs(r.other / 100 - r.y) else short[match[t]]
            add(f"| {short[t]} | {iid} | {items.loc[iid, 'locator']} | {r.self:.0f} | {r.other:.0f} | {int(r.grade)} | {closer} |")
    add("")
    add("**Scoring checks**\n")
    j = rec.dropna(subset=["judge_grade"])
    add(f"- Rule-graded {int(rec.rule_grade.notna().sum())} of {len(rec)}; judged {len(j)}; ungraded {int(rec.grade.isna().sum())}.")
    sb = rec.dropna(subset=["spec_rule_grade", "judge_grade"])
    if len(sb):
        add(f"- Spec shortcut (sim < 0.3 → 0, sim ≥ 0.9 → 3/2) agrees with the judge on {(sb.spec_rule_grade == sb.judge_grade).mean():.0%} of {len(sb)} overlapping outputs; "
            f"judge gave ≥ 2 to {int(((sb.judge_grade >= 2) & (sb.similarity < 0.3)).sum())} outputs the shortcut would have scored 0.")
    sim_q = rec.groupby("model_id").similarity.describe()[["25%", "50%", "75%"]].rename(index=short).round(2)
    add("- Similarity quartiles by model: " + "; ".join(f"{i} {r['25%']}/{r['50%']}/{r['75%']}" for i, r in sim_q.iterrows()))
    add("- Judge–human agreement (kappa) not yet measured: needs a human-graded stratified sample.")
    add("")
    add("## 9. Kill criteria and redesign triggers\n")
    for m in models:
        s = summ["models"][m]
        fl = []
        if s["c1_parse_fail"] > 0.10:
            fl.append("C1 parse failures > 10%")
        if s["c1_distinct"] <= 3 or s["c1_gist_sd"] < 5:
            fl.append("C1 confidence near-constant")
        if not 0.15 <= s["base_rate"] <= 0.85:
            fl.append(f"base rate {s['base_rate']:.2f} outside 15-85%")
        if max(s["grade_dist"]) > 0.8 * sum(s["grade_dist"]):
            fl.append(">80% of grades in one level")
        if s["c1_leaks"] > 0.05 * len(ids):
            fl.append("C1 leaks > 5%")
        if s["c1_think_max"] > 0:
            fl.append("thinking tokens present in C1")
        add(f"- {short[m]}: " + ("; ".join(fl) if fl else "ok"))
    add("")
    add("Figures: `figures/fig0_summary.png`, `fig1_base_rates.png`, `fig2_auroc_matrix.png`, `fig3_delta_forest.png`, `fig4_within_text.png`, `fig5_calibration.png`.")

    (out_dir / "report.md").write_text("\n".join(L))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summ, f, indent=2, default=lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else (float(x) if isinstance(x, (np.floating, float)) else str(x)))

    if not args.no_charts:
        plt = setup_mpl()
        fig_dir.mkdir(parents=True, exist_ok=True)
        heatmap(plt, br, fig_dir / "fig1_base_rates.png", f"Recall base rate (grade ≥ {thr}) by text and tier", 0, 1,
                note="Darker = more items recalled. Rows are the designed popularity strata.")
        heatmap(plt, mats[args.outcome].rename_axis(index="predictor", columns="target"), fig_dir / "fig2_auroc_matrix.png",
                f"AUROC: row predicts column's {args.outcome} recall", 0.5, 1.0, fmt_s="{:.2f}", outline_diag=True,
                note="Outlined diagonal = self-prediction (C1). 0.5 = chance.")
        forest(plt, forest_rows, fig_dir / "fig3_delta_forest.png", "Self-advantage: self C1 AUROC minus comparator AUROC")
        dots(plt, dot_data, fig_dir / "fig4_within_text.png", "Pooled vs within-text AUROC by predictor")
        reliability(plt, curves, fig_dir / "fig5_calibration.png", "Calibration of self-predictions (C1)")
        sdata = {}
        for t, d in dot_data.items():
            others = {k: v for k, v in d.items() if k not in ("self", "log n-gram freq")}
            bp = max(others, key=lambda k: others[k][0])
            bw = max(others, key=lambda k: -1 if np.isnan(others[k][1]) else others[k][1])
            sdata[t] = {"pooled": (d["self"][0], others[bp][0], bp.replace(" (C3)", "")),
                        "within": (d["self"][1], others[bw][1], bw.replace(" (C3)", ""))}
        summary_chart(plt, sdata, fig_dir / "fig0_summary.png")
    print("\n".join(L))


if __name__ == "__main__":
    main()
