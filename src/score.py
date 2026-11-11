"""Score pilot calls.

1. Parse C1/C3/C3self JSON predictions and C1L P(Yes).
2. Score C2 recall attempts: normalized Levenshtein similarity, ROUGE-L, critical-element accuracy.
3. Grade: rule-based level 3 (sim >= 0.90 and all critical elements) and level 0 (empty / NONEXISTENT);
   everything else goes to the LLM judge (cheap, so the pilot judges the whole middle band rather than
   only 0.3-0.9; the spec's rule band is recomputed so its shortcut can be checked against the judge).
   Refusals are flagged separately.

  python -I src/score.py [--no-judge]
Writes results/<name>/predictions.csv, recall.csv, judge.jsonl.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import prompts  # noqa: E402
from common import ROOT, read_jsonl, append_jsonl, load_config, clean_quotes, nfkc  # noqa: E402
from run_calls import call_api, load_pricing  # noqa: E402

REFUSAL_RE = re.compile(
    r"declin|copyright|I (?:don't|do not) have (?:reliable|accurate|confident|sufficient|access|the (?:exact|full) text)|"
    r"I can(?:no|')t (?:reliably|accurately|reproduce|provide|share|help)|I(?:'m| am) (?:not able|unable) to (?:provide|reproduce|recall)|"
    r"not (?:able|confident enough) to (?:reproduce|recall)|unable to (?:reproduce|recall|provide)|rather than (?:risk|guess)|as an AI", re.I)


# ---------------------------------------------------------------- normalization and metrics
def normalize(s: str) -> str:
    s = nfkc(clean_quotes(s or "")).lower()
    s = s.replace("\\left", "").replace("\\right", "").replace("\\boldsymbol", "\\mathbf")
    s = re.sub(r"(?<=\w)-(?=\w)", "", s)          # to-day -> today
    s = re.sub(r"[^\w\s\\{}^_]", " ", s)            # drop punctuation (prose pilot)
    return re.sub(r"\s+", " ", s).strip()


def strip_echo(output: str, it) -> str:
    """Remove a leading echo of the locator/title and wrapping quotes."""
    o = (output or "").strip()
    o = re.sub(r"^```\w*\s*|\s*```$", "", o)
    loc = re.escape(it["locator"])
    o = re.sub(rf"^\W*(?:{re.escape(it['item_noun'])}\s*)?{loc}\W*\n", "", o, flags=re.I)
    o = re.sub(rf"^\W*(?:{re.escape(it['item_noun'])}\s*)?{loc}[\s:.\-]+", "", o, flags=re.I)
    return o.strip().strip('"').strip()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1 - levenshtein(a, b) / max(len(a), len(b))


def rouge_l(ref_toks, hyp_toks) -> float:
    if not ref_toks or not hyp_toks:
        return 0.0
    prev = [0] * (len(hyp_toks) + 1)
    for r in ref_toks:
        cur = [0]
        for j, h in enumerate(hyp_toks, 1):
            cur.append(prev[j - 1] + 1 if r == h else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    p, rc = lcs / len(hyp_toks), lcs / len(ref_toks)
    return 0.0 if lcs == 0 else 2 * p * rc / (p + rc)


def critical_accuracy(crit, hyp_norm_toks):
    if not crit:
        return None
    pool = {}
    for t in hyp_norm_toks:
        pool[t] = pool.get(t, 0) + 1
    hit = 0
    for c in crit:
        c = normalize(c)
        if pool.get(c, 0) > 0:
            pool[c] -= 1
            hit += 1
    return hit / len(crit)


def score_recall(it, raw):
    out = strip_echo(raw, it)
    flags = dict(refusal=bool(REFUSAL_RE.search((out or "")[:400])),
                 nonexistent=out.strip().upper().startswith("NONEXISTENT"),
                 empty=not out.strip())
    ref = normalize(it["ground_truth"]).split()
    hyp_all = normalize(out).split()
    hyp = hyp_all[:len(ref)]  # prefix alignment: compare the same span length
    sim = char_sim(" ".join(ref), " ".join(hyp))
    rl = rouge_l(ref, hyp)
    ca = critical_accuracy(it["critical_elements"], hyp_all[:int(len(ref) * 1.5) + 2])
    return dict(similarity=round(sim, 4), rougeL=round(rl, 4), critical_accuracy=None if ca is None else round(ca, 4),
                **flags, cleaned_output=out)


def rule_grade(s):
    if s["empty"] or s["nonexistent"]:
        return 0
    if s["similarity"] >= 0.90 and (s["critical_accuracy"] in (None, 1.0)):
        return 3
    return None


def spec_rule_band(s):
    """The spec's shortcut: judge only 0.3 <= sim < 0.9; grade the rest by rule."""
    if s["empty"] or s["nonexistent"] or s["similarity"] < 0.3:
        return 0
    if s["similarity"] >= 0.9:
        return 3 if s["critical_accuracy"] in (None, 1.0) else 2
    return None


# ---------------------------------------------------------------- prediction parsing
def parse_pred(raw):
    if raw is None:
        return None
    m = re.search(r"\{[^{}]*\}", raw)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            d = json.loads(re.sub(r"(\w+)\s*:", r'"\1":', m.group(0)).replace('""', '"'))
        except Exception:
            return None
    out = {}
    for k in ("exists", "verbatim", "gist"):
        v = d.get(k)
        try:
            v = float(str(v).strip().rstrip("%"))
        except (TypeError, ValueError):
            return None
        if not 0 <= v <= 100:
            return None
        out[k] = v
    return out


def p_yes(top):
    if not top:
        return None
    y = n = 0.0
    for t in top:
        tok = (t.get("token") or "").strip().lower()
        p = math.exp(t.get("logprob", -99))
        if tok in ("yes", "y"):
            y += p
        elif tok in ("no", "n"):
            n += p
    return None if y + n == 0 else y / (y + n)


# ---------------------------------------------------------------- judge
def parse_judge(raw):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        g = int(json.loads(m.group(0))["grade"]) if m else None
    except Exception:
        m2 = re.search(r'"grade"\s*:\s*([0-3])', raw)
        g = int(m2.group(1)) if m2 else None
    return g if g in (0, 1, 2, 3) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / cfg["out_dir"]
    items = {it["item_id"]: it for it in read_jsonl(ROOT / cfg["items"])}
    calls = [c for c in read_jsonl(out_dir / "calls.jsonl") if c.get("raw_output") is not None or c["condition"] == "C1L"]
    # keep the latest successful row per key
    latest = {}
    for c in calls:
        latest[(c["item_id"], c["model_id"], c["target_id"], c["condition"])] = c
    calls = list(latest.values())

    # predictions
    pred_rows = []
    for c in calls:
        if c["condition"] in ("C1", "C3", "C3self"):
            p = parse_pred(c.get("raw_output"))
            raw = c.get("raw_output") or ""
            extra = len(re.sub(r"```(?:json)?|\{[^{}]*\}", "", raw).strip())  # text outside the JSON (C1 leak check)
            pred_rows.append(dict(item_id=c["item_id"], predictor=c["model_id"], target=c["target_id"],
                                  condition=c["condition"], prompt_variant=c["prompt_variant"], parse_ok=p is not None,
                                  exists=p and p["exists"], verbatim=p and p["verbatim"], gist=p and p["gist"],
                                  extra_text_chars=extra, leak=extra > 40,
                                  thinking_tokens=(c.get("usage") or {}).get("thinking_tokens")))
        elif c["condition"] == "C1L":
            pred_rows.append(dict(item_id=c["item_id"], predictor=c["model_id"], target=c["target_id"],
                                  condition="C1L", prompt_variant=c["prompt_variant"],
                                  parse_ok=p_yes(c.get("top_logprobs")) is not None,
                                  exists=None, verbatim=None, gist=(lambda v: None if v is None else 100 * v)(p_yes(c.get("top_logprobs"))),
                                  extra_text_chars=0, leak=False, thinking_tokens=None))
    write_csv(out_dir / "predictions.csv", pred_rows)

    # recall scoring
    rec = []
    for c in calls:
        if c["condition"] != "C2":
            continue
        it = items[c["item_id"]]
        s = score_recall(it, c["raw_output"])
        rec.append(dict(item_id=c["item_id"], model_id=c["model_id"], text_id=it["text_id"],
                        popularity_tier=it["popularity_tier"], finish_reason=c.get("finish_reason"), **s,
                        rule_grade=rule_grade(s), spec_rule_grade=spec_rule_band(s)))

    # judge everything not rule-graded (refusals included, so the judge's 0 is recorded)
    judge_path = out_dir / "judge.jsonl"
    cached = {(r["item_id"], r["model_id"]): r for r in read_jsonl(judge_path) if r.get("grade") is not None}
    if not args.no_judge:
        pricing = load_pricing(cfg)
        todo = [r for r in rec if r["rule_grade"] is None and (r["item_id"], r["model_id"]) not in cached]
        print(f"judging {len(todo)} outputs with {cfg['judge']['id']}")

        def do(r):
            it = items[r["item_id"]]
            sys_p, user_p = prompts.build("JUDGE", it, "a", output=r["cleaned_output"])
            extra = {"reasoning": cfg["judge"]["reasoning"]} if cfg["judge"].get("reasoning") else None
            d, _, _ = call_api(cfg, pricing, cfg["judge"]["id"], sys_p, user_p, cfg["max_tokens"]["JUDGE"], extra)
            raw = None if "error" in d else d["choices"][0]["message"].get("content")
            row = dict(item_id=r["item_id"], model_id=r["model_id"], judge=cfg["judge"]["id"], raw=raw,
                       grade=parse_judge(raw), usage=d.get("usage"), error=d.get("error"))
            append_jsonl(judge_path, row)
            return row

        with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as ex:
            for row in ex.map(do, todo):
                if row["grade"] is not None:
                    cached[(row["item_id"], row["model_id"])] = row
    for r in rec:
        j = cached.get((r["item_id"], r["model_id"]))
        r["judge_grade"] = j["grade"] if j else None
        r["grade"] = r["rule_grade"] if r["rule_grade"] is not None else r["judge_grade"]
        r.pop("cleaned_output")
    write_csv(out_dir / "recall.csv", rec)
    n_pf = sum(1 for p in pred_rows if not p["parse_ok"])
    print(f"{len(pred_rows)} predictions ({n_pf} parse failures), {len(rec)} recall attempts, "
          f"{sum(r['grade'] is None for r in rec)} ungraded")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
