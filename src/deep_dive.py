"""Second-pass analyses that remove the shared item-difficulty signal and look for model-specific knowledge.

  A private signal      self minus outsider consensus vs outcome (within-text residualized)
  B relative knowledge  on items where two models' outcomes differ, does each know which side it is on?
  C projection          does an outsider's prediction about Q track Q's outcome or the outsider's own?
  D address errors      failed outputs that reproduce a *different* unit of the same text
  E ratios              failure-flag rate, success-endorse rate, implied P(verbatim | gist) vs observed
  F continuous outcome  Spearman(confidence, similarity), pooled and within text
  G target-specific     within-item deviations of a predictor's 3 predictions vs within-item outcome deviations

  python -I src/deep_dive.py --config pilot_claude
Writes results/<name>/deep_dive.md and address_errors.csv.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, read_jsonl, load_config, first_n_words  # noqa: E402
from analyze import auroc, fmt  # noqa: E402
import build_items as bi  # noqa: E402
from score import normalize, strip_echo, char_sim  # noqa: E402

B = 2000


def rankdata(x):
    return pd.Series(x).rank().to_numpy()


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5:
        return np.nan
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def demean_by(x, g):
    x = pd.Series(np.asarray(x, float))
    return (x - x.groupby(np.asarray(g)).transform("mean")).to_numpy()


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def boot_ci(f, n, rng, strata=None):
    vals = []
    groups = None if strata is None else [np.where(strata == g)[0] for g in pd.unique(strata)]
    for _ in range(B):
        ix = rng.integers(0, n, n) if groups is None else np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        v = f(ix)
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < B / 2:
        return np.nan, np.nan
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


# ---------------------------------------------------------------- unit inventories for address-error detection
def unit_inventory():
    inv = {}
    inv["kjv"] = {f"{b} {c}:{v}": t for b, c, v, t in bi.parse_kjv()}
    inv["sonnets"] = {str(k): "\n".join(lines[:4]) for k, lines in bi.parse_sonnets().items()}
    inv["tractatus"] = {k: first_n_words(t, bi.MAX_WORDS) for k, (t, _) in bi.parse_tractatus().items()}
    inv["us_constitution"] = {f"Article {bi.ROMAN[a]}" + (f", Section {s}" if s is not None else ""): first_n_words(t, bi.MAX_WORDS)
                              for a, s, t in bi.parse_us()}
    inv["tn_constitution"] = {f"Article {bi.ROMAN[a]}, Section {s}": first_n_words(t, bi.MAX_WORDS) for a, s, t in bi.parse_tn()}
    return inv


def best_other_unit(output, it, inv_text, norm_cache):
    hyp = normalize(strip_echo(output, it)).split()
    if not hyp:
        return None, 0.0
    hs = set(hyp)
    # prefilter by word overlap, then exact char similarity on the top candidates
    scored = []
    for loc, toks in norm_cache.items():
        if loc == it["locator"]:
            continue
        ov = len(hs & toks[1]) / (len(toks[1]) + 1e-9)
        scored.append((ov, loc))
    scored.sort(reverse=True)
    best = (None, 0.0)
    for _, loc in scored[:15]:
        ref = norm_cache[loc][0]
        s = char_sim(" ".join(ref), " ".join(hyp[:len(ref)]))
        if s > best[1]:
            best = (loc, s)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot_claude")
    ap.add_argument("--addr-threshold", type=float, default=0.75)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / cfg["out_dir"]
    rng = np.random.default_rng(1)
    items = pd.DataFrame(read_jsonl(ROOT / cfg["items"])).set_index("item_id")
    ids = list(items.index)
    text = items.text_id.reindex(ids).to_numpy()
    models = [m["id"] for m in cfg["models"]]
    sh = {m: m.replace("claude-", "") for m in models}
    pred = pd.read_csv(out_dir / "predictions.csv")
    rec = pd.read_csv(out_dir / "recall.csv")
    rec = rec[rec.model_id.isin(models)]
    Y = rec.assign(y=(rec.grade >= 2).astype(float)).pivot_table(index="item_id", columns="model_id", values="y").reindex(ids)
    V = rec.assign(v=(rec.grade >= 3).astype(float)).pivot_table(index="item_id", columns="model_id", values="v").reindex(ids)
    SIM = rec.pivot_table(index="item_id", columns="model_id", values="similarity").reindex(ids)

    def C(p, t, field="gist"):
        cond = "C1" if p == t else "C3"
        sub = pred[(pred.predictor == p) & (pred.target == t) & (pred.condition == cond)]
        return sub.set_index("item_id")[field].reindex(ids).astype(float).to_numpy() / 100

    L = []
    add = L.append
    add(f"# Deep dive: {cfg['name']}\n")
    add("Every analysis here removes, or conditions away, the part of confidence that is shared item difficulty, "
        "and asks whether what is left carries model-specific knowledge. n = 120 items; treat single numbers as indicative.\n")

    # ---------------------------------------------------------------- A private signal
    add("## A. Private signal: is the model right when it disagrees with the outsiders about itself?\n")
    add("private = self C1 − mean of the two siblings' C3 about the same target. Both private signal and outcome are demeaned within text "
        "(so 'this text is easy' is removed), then correlated. Positive = when M is more optimistic than outsiders about itself, it tends to be right. "
        "Control row: the same statistic for each sibling's deviation from the other two (self + other sibling).\n")
    add("| target | signal | n | within-text r with outcome | 95% CI | AUROC of raw signal |")
    add("|---|---|---|---|---|---|")
    for t in models:
        y = Y[t].to_numpy()
        others = [p for p in models if p != t]
        preds = {p: C(p, t) for p in models}
        rows = [("self − outsiders", preds[t] - np.nanmean([preds[p] for p in others], axis=0))]
        for q in others:
            rest = [p for p in models if p != q]
            rows.append((f"{sh[q]} − rest", preds[q] - np.nanmean([preds[p] for p in rest], axis=0)))
        for lab, sig in rows:
            f = lambda ix, sig=sig: corr(demean_by(sig[ix], text[ix]), demean_by(y[ix], text[ix]))
            r = f(np.arange(len(y)))
            lo, hi = boot_ci(f, len(y), rng, strata=text)
            add(f"| {sh[t]} | {lab} | {int((~np.isnan(sig) & ~np.isnan(y)).sum())} | {fmt(r)} | [{fmt(lo)}, {fmt(hi)}] | {fmt(auroc(sig, y))} |")
    add("")

    # ---------------------------------------------------------------- B relative knowledge
    add("## B. Relative self-knowledge on discordant items\n")
    add("Items where P and Q got different outcomes. P's view: C1_P − C3_{P→Q} (how much better P expects to do than Q). "
        "Q's view of the same contrast: C3_{Q→P} − C1_Q. AUROC for predicting 'P succeeded, Q failed'. 0.5 = no idea who wins.\n")
    add("| P | Q | discordant n (P won / Q won) | P's view AUROC | Q's view AUROC | P views itself better? |")
    add("|---|---|---|---|---|---|")
    for i, p in enumerate(models):
        for q in models[i + 1:]:
            yp, yq = Y[p].to_numpy(), Y[q].to_numpy()
            d = (yp != yq) & ~np.isnan(yp) & ~np.isnan(yq)
            lab = (yp[d] > yq[d]).astype(float)
            pv = (C(p, p) - C(p, q))[d]
            qv = (C(q, p) - C(q, q))[d]
            a_p, a_q = auroc(pv, lab), auroc(qv, lab)
            add(f"| {sh[p]} | {sh[q]} | {int(d.sum())} ({int(lab.sum())}/{int(len(lab) - lab.sum())}) | {fmt(a_p)} | {fmt(a_q)} | "
                f"{'yes' if a_p > a_q else 'no'} |")
    add("")

    # ---------------------------------------------------------------- C projection
    add("## C. Projection: whose recall does an outsider's prediction track?\n")
    add("On items where predictor P and target Q have different outcomes, AUROC of P's prediction about Q against Q's outcome. "
        "Below 0.5 means P's prediction about Q follows P's *own* outcome: P assumes Q knows what P knows.\n")
    add("| predictor P | target Q | discordant n | AUROC vs Q's outcome | reading |")
    add("|---|---|---|---|---|")
    for p in models:
        for q in models:
            if p == q:
                continue
            yp, yq = Y[p].to_numpy(), Y[q].to_numpy()
            d = (yp != yq) & ~np.isnan(yp) & ~np.isnan(yq)
            a = auroc(C(p, q)[d], yq[d])
            reading = "tracks Q" if a > 0.6 else ("projects own knowledge" if a < 0.4 else "no signal")
            add(f"| {sh[p]} | {sh[q]} | {int(d.sum())} | {fmt(a)} | {reading} |")
    add("")

    add("**Nested version** (knowledge turned out nested, so most pairs have one-sided discordance). Mean stated gist about the weaker model Q, "
        "split by outcome cell. If an outsider P projects, its numbers about Q stay high on items only P recalled.\n")
    add("| stronger P | weaker Q | predictor | both recalled | only P recalled | neither | drop (both → only P) |")
    add("|---|---|---|---|---|---|---|")
    for p_, q_ in ((models[0], models[2]), (models[1], models[2]), (models[0], models[1])):
        yp, yq = Y[p_].to_numpy(), Y[q_].to_numpy()
        cells = {"both": (yp == 1) & (yq == 1), "onlyP": (yp == 1) & (yq == 0), "neither": (yp == 0) & (yq == 0)}
        for who in models:
            s_ = C(who, q_)
            m_ = {k: np.nanmean(s_[v]) if v.any() else np.nan for k, v in cells.items()}
            add(f"| {sh[p_]} | {sh[q_]} | {sh[who]}{' (self)' if who == q_ else ''} | {fmt(m_['both'], 2)} ({int(cells['both'].sum())}) | "
                f"{fmt(m_['onlyP'], 2)} ({int(cells['onlyP'].sum())}) | {fmt(m_['neither'], 2)} ({int(cells['neither'].sum())}) | {fmt(m_['both'] - m_['onlyP'], 2)} |")
    add("")

    # ---------------------------------------------------------------- D address errors
    add("## D. Address errors: right text, wrong unit\n")
    inv = unit_inventory()
    cache = {tx: {loc: (normalize(t).split(), set(normalize(t).split())) for loc, t in d.items()} for tx, d in inv.items()}
    calls = [json.loads(line) for line in open(out_dir / "calls.jsonl")]
    c2 = {(r["item_id"], r["model_id"]): r["raw_output"] for r in calls if r["condition"] == "C2" and r.get("raw_output")}
    addr_rows = []
    gi = {it["item_id"]: it for it in read_jsonl(ROOT / cfg["items"])}
    for _, r in rec.iterrows():
        if r.grade >= 2:
            continue
        it = gi[r.item_id]
        out = c2.get((r.item_id, r.model_id))
        if not out:
            continue
        loc, s = best_other_unit(out, it, inv[it["text_id"]], cache[it["text_id"]])
        addr_rows.append(dict(item_id=r.item_id, model_id=r.model_id, text_id=it["text_id"], asked=it["locator"],
                              produced=loc, sim_to_produced=round(s, 3), own_sim=r.similarity, address_error=s >= args.addr_threshold))
    ad = pd.DataFrame(addr_rows)
    ad.to_csv(out_dir / "address_errors.csv", index=False)
    if len(ad):
        add(f"A failed output (grade < 2) is an address error if it matches another unit of the same text with similarity ≥ {args.addr_threshold}.\n")
        tab = ad.groupby(["model_id", "text_id"]).agg(failures=("address_error", "size"), address_errors=("address_error", "sum")).reset_index()
        tab["model_id"] = tab.model_id.map(sh)
        add(tab.to_markdown(index=False))
        add("")
        ex = ad[ad.address_error].sort_values("sim_to_produced", ascending=False).head(8)
        if len(ex):
            add("Examples: " + "; ".join(f"{sh[r.model_id]} asked {r.asked} → gave {r.produced} ({r.sim_to_produced:.2f})" for r in ex.itertuples()) + "\n")
        add("Does confidence track *content* (knows some unit of the text verbatim, address-error counted as known) or *address* (correct unit)?\n")
        add("| model | address errors | self AUROC vs correct unit | self AUROC vs content known | mean stated gist on address errors | on true failures |")
        add("|---|---|---|---|---|---|")
        for m in models:
            y = Y[m].to_numpy()
            ae = ad[(ad.model_id == m) & ad.address_error].item_id
            content = y.copy()
            mask = np.isin(ids, ae)
            content[mask] = 1
            s = C(m, m)
            fail_true = (y == 0) & ~mask
            add(f"| {sh[m]} | {int(mask.sum())} | {fmt(auroc(s, y))} | {fmt(auroc(s, content))} | "
                f"{fmt(np.nanmean(s[mask]) if mask.any() else np.nan, 2)} | {fmt(np.nanmean(s[fail_true]), 2)} |")
        add("")

    # ---------------------------------------------------------------- E ratios
    add("## E. Ratios: flagging failures and endorsing successes\n")
    add("Failure-flag rate = share of the target's failures that the predictor rated below 50%. "
        "Success-endorse rate = share of successes rated 50% or above. Implied P(verbatim | gist) = mean(verbatim)/mean(gist) "
        "of the predictor's numbers; observed = share of gist-correct attempts that were verbatim.\n")
    add("| target | predictor | failures flagged | successes endorsed | implied P(verb | gist) | observed P(verb | gist) |")
    add("|---|---|---|---|---|---|")
    for t in models:
        y, v = Y[t].to_numpy(), V[t].to_numpy()
        obs = np.nansum(v * (y == 1)) / np.nansum(y == 1)
        for p in models:
            g, vb = C(p, t), C(p, t, "verbatim")
            ff = np.nanmean(g[y == 0] < 0.5) if (y == 0).any() else np.nan
            se = np.nanmean(g[y == 1] >= 0.5)
            add(f"| {sh[t]} | {sh[p]}{' (self)' if p == t else ''} | {fmt(ff, 2)} ({int((y == 0).sum())}) | {fmt(se, 2)} | "
                f"{fmt(np.nanmean(vb) / np.nanmean(g), 2)} | {fmt(obs, 2)} |")
    add("")

    # ---------------------------------------------------------------- F continuous
    add("## F. Continuous outcome: Spearman(stated gist, similarity)\n")
    add("Uses the full similarity score instead of the 0/1 grade. Within-text = mean of per-text Spearman (texts with variance).\n")
    add("| target | predictor | pooled ρ | within-text ρ |")
    add("|---|---|---|---|")
    for t in models:
        sim = SIM[t].to_numpy()
        for p in models:
            s = C(p, t)
            w = [spearman(s[text == tx], sim[text == tx]) for tx in pd.unique(text)]
            w = [x for x in w if not np.isnan(x)]
            add(f"| {sh[t]} | {sh[p]}{' (self)' if p == t else ''} | {fmt(spearman(s, sim))} | {fmt(np.mean(w) if w else np.nan)} |")
    add("")

    # ---------------------------------------------------------------- G target-specific
    add("## G. Target-specific prediction: does a predictor know *which model* will get a given item?\n")
    add("For each item, take predictor P's three numbers (about Opus, Sonnet, Haiku) and subtract their mean; do the same for the three "
        "actual outcomes. Correlation across all item × target cells. Item difficulty cancels exactly; only 'who will get this one' remains. "
        "Also: share of P's prediction variance that is item-level (shared across targets).\n")
    add("| predictor | r(within-item prediction deviation, within-item outcome deviation) | 95% CI | item-level share of prediction variance |")
    add("|---|---|---|---|")
    Ymat = np.column_stack([Y[t].to_numpy() for t in models])
    ydev = Ymat - np.nanmean(Ymat, axis=1, keepdims=True)
    for p in models:
        P = np.column_stack([C(p, t) for t in models])
        pdev = P - np.nanmean(P, axis=1, keepdims=True)
        f = lambda ix: corr(pdev[ix].ravel(), ydev[ix].ravel())
        r = f(np.arange(len(ids)))
        lo, hi = boot_ci(f, len(ids), rng)
        tot = np.nanvar(P)
        item_share = np.nanvar(np.nanmean(P, axis=1)) / tot if tot else np.nan
        add(f"| {sh[p]} | {fmt(r)} | [{fmt(lo)}, {fmt(hi)}] | {fmt(item_share, 2)} |")
    add("")
    add("**Double-centered** (also removes each target's overall level, e.g. 'Haiku is weaker'): only item × target interaction remains.\n")
    add("| predictor | r(interaction) | 95% CI | mean stated gist: self vs about others |")
    add("|---|---|---|---|")
    def dc(M):
        return M - np.nanmean(M, axis=1, keepdims=True) - np.nanmean(M, axis=0, keepdims=True) + np.nanmean(M)
    for k, p in enumerate(models):
        P = np.column_stack([C(p, t) for t in models])
        f = lambda ix: corr(dc(P[ix]).ravel(), dc(Ymat[ix]).ravel())
        r = f(np.arange(len(ids)))
        lo, hi = boot_ci(f, len(ids), rng)
        others_mean = np.nanmean(np.delete(P, k, axis=1))
        add(f"| {sh[p]} | {fmt(r)} | [{fmt(lo)}, {fmt(hi)}] | {fmt(np.nanmean(P[:, k]), 2)} vs {fmt(others_mean, 2)} |")
    add("")
    (out_dir / "deep_dive.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
