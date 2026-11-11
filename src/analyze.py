"""Pilot analysis: feasibility checks, AUROC matrix, self-advantage with paired bootstrap, framing control,
and outsider baselines the spec does not include (popularity, other models' actual recall).

  python -I src/analyze.py
Writes results/<name>/report.md and summary.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, read_jsonl, load_config  # noqa: E402

B = 2000


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    ok = ~np.isnan(score)
    score, y = score[ok], y[ok]
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(score).rank().to_numpy()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def paired_boot(s_a, s_b, y, rng):
    s_a, s_b, y = map(lambda a: np.asarray(a, float), (s_a, s_b, y))
    n = len(y)
    d = []
    for _ in range(B):
        ix = rng.integers(0, n, n)
        a, b = auroc(s_a[ix], y[ix]), auroc(s_b[ix], y[ix])
        if not (np.isnan(a) or np.isnan(b)):
            d.append(a - b)
    if len(d) < B / 2:
        return np.nan, np.nan, np.nan
    d = np.array(d)
    return np.percentile(d, 2.5), np.percentile(d, 97.5), (d <= 0).mean()


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot")
    ap.add_argument("--outcome", default="gist", choices=["gist", "verbatim"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / cfg["out_dir"]
    rng = np.random.default_rng(0)
    items = pd.DataFrame(read_jsonl(ROOT / cfg["items"])).set_index("item_id")
    pop_path = ROOT / "data" / "items" / "pilot_popularity.csv"
    if pop_path.exists():
        items = items.join(pd.read_csv(pop_path).set_index("item_id")[["log_freq"]])
    pred = pd.read_csv(out_dir / "predictions.csv")
    rec = pd.read_csv(out_dir / "recall.csv")
    models = [m["id"] for m in cfg["models"]]
    short = {m["id"]: m["id"].split("/")[-1] for m in cfg["models"]}
    match = {m["id"]: m["match"] for m in cfg["models"]}
    family = {m["id"]: m["family"] for m in cfg["models"]}
    thr = 2 if args.outcome == "gist" else 3
    col = args.outcome
    ids = list(items.index)

    rec["y"] = (rec["grade"].fillna(0) >= thr).astype(int)
    Y = rec.pivot_table(index="item_id", columns="model_id", values="y").reindex(ids)
    G = rec.pivot_table(index="item_id", columns="model_id", values="grade").reindex(ids)

    def score(predictor, target, cond, field=col):
        sub = pred[(pred.predictor == predictor) & (pred.target == target) & (pred.condition == cond)]
        f = "gist" if cond == "C1L" else field
        return sub.set_index("item_id")[f].reindex(ids).astype(float)

    L = []  # report lines
    summary = {"outcome": args.outcome, "n_items": len(ids), "models": {}}

    # ---- 1. feasibility
    L.append(f"# Pilot report (outcome: {args.outcome}, grade >= {thr})\n")
    L.append(f"{len(ids)} items, {len(models)} models. Judge: {cfg['judge']['id']}.\n")
    L.append("## 1. Feasibility checks\n")
    L.append("| model | base rate | grade 0/1/2/3 | refusals | C1 parse fail | C3 parse fail | C1 gist SD | C1 distinct values | C1L parse fail |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in models:
        r = rec[rec.model_id == m]
        dist = [int((r.grade == g).sum()) for g in range(4)]
        c1 = pred[(pred.predictor == m) & (pred.condition == "C1")]
        c3 = pred[(pred.predictor == m) & (pred.condition == "C3")]
        c1l = pred[(pred.predictor == m) & (pred.condition == "C1L")]
        br = r.y.mean()
        summary["models"][m] = dict(base_rate=br, grade_dist=dist, refusals=int(r.refusal.sum()),
                                    c1_parse_fail=1 - c1.parse_ok.mean() if len(c1) else None,
                                    c3_parse_fail=1 - c3.parse_ok.mean() if len(c3) else None,
                                    c1_gist_sd=c1[col].std(), c1_distinct=int(c1[col].nunique()))
        L.append(f"| {short[m]} | {fmt(br, 2)} | {'/'.join(map(str, dist))} | {int(r.refusal.sum())} | "
                 f"{fmt(1 - c1.parse_ok.mean() if len(c1) else np.nan, 2)} | {fmt(1 - c3.parse_ok.mean() if len(c3) else np.nan, 2)} | "
                 f"{fmt(c1[col].std(), 1)} | {c1[col].nunique()} | {fmt(1 - c1l.parse_ok.mean() if len(c1l) else np.nan, 2)} |")
    L.append("")
    L.append("Base rate by text (rows) and model (columns):\n")
    bt = rec.pivot_table(index="text_id", columns="model_id", values="y", aggfunc="mean").rename(columns=short)
    L.append(bt.round(2).to_markdown())
    L.append("")

    # ---- 2. AUROC matrix
    L.append(f"## 2. AUROC matrix: predictor (rows) predicting target's {args.outcome} recall (columns)\n")
    L.append("Diagonal = self-prediction (C1). Off-diagonal = cross-prediction (C3).\n")
    mat = pd.DataFrame(index=[short[m] for m in models], columns=[short[m] for m in models], dtype=float)
    for p in models:
        for t in models:
            s = score(p, t, "C1" if p == t else "C3")
            mat.loc[short[p], short[t]] = auroc(s, Y[t])
    L.append(mat.round(3).to_markdown())
    summary["auroc_matrix"] = mat.round(4).to_dict()
    L.append("")

    # ---- 3. self-advantage
    L.append("## 3. Self-advantage (H1): Δ = AUROC(M→M, C1) − AUROC(matched→M, C3), paired bootstrap 95% CI\n")
    L.append("| target | self C1 | matched | matched AUROC | Δ | 95% CI | P(Δ≤0) | same-family other size | best other | self C1L (P(Yes)) | self C3-framed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in models:
        y = Y[t]
        s_self = score(t, t, "C1")
        s_m = score(match[t], t, "C3")
        a_self, a_m = auroc(s_self, y), auroc(s_m, y)
        both = ~(s_self.isna() | s_m.isna() | y.isna())
        lo, hi, p0 = paired_boot(s_self[both], s_m[both], y[both], rng)
        fam = [p for p in models if p != t and family[p] == family[t]]
        a_fam = auroc(score(fam[0], t, "C3"), y) if fam else np.nan
        others = {p: auroc(score(p, t, "C3"), y) for p in models if p != t}
        best = max(others, key=lambda k: -1 if np.isnan(others[k]) else others[k])
        a_c1l = auroc(score(t, t, "C1L"), y)
        a_c3s = auroc(score(t, t, "C3self"), y)
        summary["models"][t].update(self_auroc=a_self, matched_auroc=a_m, delta=a_self - a_m, delta_ci=[lo, hi],
                                    p_delta_le_0=p0, same_family_auroc=a_fam, c1l_auroc=a_c1l, c3self_auroc=a_c3s)
        L.append(f"| {short[t]} | {fmt(a_self)} | {short[match[t]]} | {fmt(a_m)} | {fmt(a_self - a_m)} | "
                 f"[{fmt(lo)}, {fmt(hi)}] | {fmt(p0, 2)} | {fmt(a_fam)} | {short[best]} {fmt(others[best])} | "
                 f"{fmt(a_c1l)} | {fmt(a_c3s)} |")
    L.append("")
    L.append("H2 (pilot can only show direction, 2 sizes per family): Δ by family and size\n")
    for fam in sorted(set(family.values())):
        ms = sorted([m for m in cfg["models"] if m["family"] == fam], key=lambda m: m["params_b"])
        L.append("- " + fam + ": " + ", ".join(f"{m['params_b']}B Δ={fmt(summary['models'][m['id']]['delta'])}" for m in ms))
    L.append("")

    # ---- 4. outsider baselines not in the spec
    L.append("## 4. Stronger outsider baselines (not in the spec)\n")
    L.append("If self-prediction does not beat these, the self-report carries no information beyond item difficulty.\n")
    L.append("| target | self C1 | popularity tier | log n-gram freq | other models' mean actual grade | best single other model's actual grade |")
    L.append("|---|---|---|---|---|---|")
    tier_score = items["popularity_tier"].map({"famous": 1, "shallow": 1, "random": 0, "deep": 0, "obscure": 0}).reindex(ids).astype(float)
    for t in models:
        y = Y[t]
        others = [p for p in models if p != t]
        a_lf = auroc(items["log_freq"].reindex(ids), y) if "log_freq" in items else np.nan
        a_cons = auroc(G[others].mean(axis=1), y)
        a_best = max(auroc(G[p], y) for p in others)
        summary["models"][t].update(tier_auroc=auroc(tier_score, y), logfreq_auroc=a_lf, consensus_auroc=a_cons)
        L.append(f"| {short[t]} | {fmt(summary['models'][t]['self_auroc'])} | {fmt(auroc(tier_score, y))} | {fmt(a_lf)} | {fmt(a_cons)} | {fmt(a_best)} |")
    L.append("")

    # ---- 5. scoring checks
    L.append("## 5. Scoring checks\n")
    j = rec.dropna(subset=["judge_grade"])
    L.append(f"- Rule-graded: {int(rec.rule_grade.notna().sum())} of {len(rec)}; judged: {len(j)}; ungraded: {int(rec.grade.isna().sum())}.")
    sb = rec.dropna(subset=["spec_rule_grade", "judge_grade"])
    if len(sb):
        agree = (sb.spec_rule_grade == sb.judge_grade).mean()
        L.append(f"- Spec shortcut (sim < 0.3 → 0, sim ≥ 0.9 → 3/2) agrees with the judge on {agree:.0%} of {len(sb)} outputs the judge also graded. "
                 f"Judge grade ≥ 2 among sim < 0.3: {int(((sb.judge_grade >= 2) & (sb.similarity < 0.3)).sum())}.")
    L.append("- Human grading of a stratified sample is still required for judge-human kappa (not automated).")
    L.append("")

    # ---- 6. kill criteria
    L.append("## 6. Kill criteria and redesign triggers\n")
    for m in models:
        s = summary["models"][m]
        flags = []
        if s["c1_parse_fail"] and s["c1_parse_fail"] > 0.10:
            flags.append("C1 parse failures > 10%")
        if s["c1_distinct"] <= 3 or (s["c1_gist_sd"] is not None and s["c1_gist_sd"] < 5):
            flags.append("C1 confidence near-constant")
        if not 0.15 <= s["base_rate"] <= 0.85:
            flags.append(f"base rate {s['base_rate']:.2f} outside 15-85%")
        if max(s["grade_dist"]) > 0.8 * sum(s["grade_dist"]):
            flags.append(">80% of grades in one level")
        L.append(f"- {short[m]}: " + ("; ".join(flags) if flags else "ok"))
    L.append("")
    (out_dir / "report.md").write_text("\n".join(L))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else float(x))
    print("\n".join(L))


if __name__ == "__main__":
    main()
