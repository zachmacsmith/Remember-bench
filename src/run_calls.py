"""Run pilot conditions C1, C1L, C3, C3self, C2 through OpenRouter (OpenAI-compatible API).

  python -I src/run_calls.py --dry-run            # count calls and estimate cost, no network
  python -I src/run_calls.py --limit-items 8      # smoke test on 8 items
  python -I src/run_calls.py                      # full pilot (resumes; skips calls already logged)

Needs OPENROUTER_API_KEY in the environment (in the cloud session the proxy can inject it).
Every call is appended to results/<name>/calls.jsonl with raw output, provider, usage and cost.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import prompts  # noqa: E402
from common import ROOT, read_jsonl, append_jsonl, load_config  # noqa: E402

PHASE_ORDER = ["C1", "C1L", "C3", "C3self", "C2"]
_lock = threading.Lock()


def load_pricing(cfg):
    with open(ROOT / cfg["pricing_snapshot"]) as f:
        return {m["id"]: m for m in json.load(f)["data"]}


def build_jobs(cfg, items):
    models = cfg["models"]
    jobs = []
    for cond in PHASE_ORDER:
        if cond not in cfg["conditions"]:
            continue
        for it in items:
            v = it["prompt_variant"]
            for m in models:
                if cond == "C3":
                    for t in models:
                        if t["id"] == m["id"]:
                            continue
                        sys_p, user_p = prompts.build("C3", it, v, target_name=t["name"])
                        jobs.append(dict(item_id=it["item_id"], model_id=m["id"], target_id=t["id"], condition="C3",
                                         prompt_variant=v, system=sys_p, user=user_p))
                    continue
                target = m["id"]
                kw = {"target_name": m["name"]} if cond == "C3self" else {}
                sys_p, user_p = prompts.build(cond, it, v, **kw)
                jobs.append(dict(item_id=it["item_id"], model_id=m["id"], target_id=target, condition=cond,
                                 prompt_variant=v, system=sys_p, user=user_p))
    return jobs


def job_key(j):
    return (j["item_id"], j["model_id"], j["target_id"], j["condition"])


def est_tokens(s):
    return len(s) / 3.8 + 4


def estimate(cfg, jobs, items, pricing, out_tokens_c2=None):
    by_item = {it["item_id"]: it for it in items}
    rows = {}
    for j in jobs:
        p = pricing[j["model_id"]]["pricing"]
        tin = est_tokens(j["system"]) + est_tokens(j["user"]) + 8
        if j["condition"] == "C2":
            tout = min(cfg["max_tokens"]["C2"], 1.5 * by_item[j["item_id"]]["length_words"] * 1.35 + 10)
        elif j["condition"] == "C1L":
            tout = 1
        else:
            tout = 25
        c = tin * float(p["prompt"]) + tout * float(p["completion"])
        r = rows.setdefault((j["model_id"], j["condition"]), [0, 0.0, 0.0, 0.0])
        r[0] += 1
        r[1] += tin
        r[2] += tout
        r[3] += c
    # judge: one call per C2 output
    jid = cfg["judge"]["id"]
    jp = pricing[jid]["pricing"]
    n_c2 = sum(1 for j in jobs if j["condition"] == "C2")
    avg_gt = sum(it["length_words"] for it in items) / len(items)
    tin = est_tokens(prompts.JUDGE) + 2 * avg_gt * 1.35 + 60
    rows[(jid, "JUDGE")] = [n_c2, n_c2 * tin, n_c2 * 30, n_c2 * (tin * float(jp["prompt"]) + 30 * float(jp["completion"]))]
    return rows


def print_estimate(rows):
    tot = [0, 0, 0, 0.0]
    print(f"{'model':42s} {'cond':7s} {'calls':>6s} {'in_tok':>9s} {'out_tok':>8s} {'USD':>8s}")
    for (m, c), (n, ti, to, usd) in sorted(rows.items()):
        print(f"{m:42s} {c:7s} {n:6d} {ti:9.0f} {to:8.0f} {usd:8.4f}")
        tot = [tot[0] + n, tot[1] + ti, tot[2] + to, tot[3] + usd]
    print(f"{'TOTAL':42s} {'':7s} {tot[0]:6d} {tot[1]:9.0f} {tot[2]:8.0f} {tot[3]:8.4f}")
    return tot


def call_api(cfg, pricing, model_id, system, user, max_tokens, extra=None, retries=6):
    import requests
    key = os.environ.get("OPENROUTER_API_KEY", "proxy-injected")
    sp = set(pricing.get(model_id, {}).get("supported_parameters", []))
    body = {"model": model_id, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens, "usage": {"include": True}}
    if "temperature" in sp:
        body["temperature"] = cfg["temperature"]
    if "seed" in sp:
        body["seed"] = cfg["seed"]
    if extra:
        body.update(extra)
    delay = 2
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = requests.post(f"{cfg['api_base']}/chat/completions", json=body, timeout=120,
                              headers={"Authorization": f"Bearer {key}", "X-Title": "remember-bench"})
            if r.status_code == 200:
                d = r.json()
                if "choices" in d and d["choices"]:
                    return d, time.time() - t0, body
                err = d.get("error", d)
            else:
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 403, 404):
                    return {"error": err}, time.time() - t0, body
        except Exception as e:  # network errors
            err = repr(e)
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return {"error": str(err)}, 0.0, body


def run_job(cfg, pricing, j, out_path):
    extra = None
    if j["condition"] == "C1L":
        extra = {"logprobs": True, "top_logprobs": 10}
    d, latency, body = call_api(cfg, pricing, j["model_id"], j["system"], j["user"], cfg["max_tokens"][j["condition"]], extra)
    row = {k: j[k] for k in ("item_id", "model_id", "target_id", "condition", "prompt_variant")}
    row.update(temperature=body.get("temperature"), seed=body.get("seed"), thinking_mode="off",
               max_tokens=body["max_tokens"], timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               latency_s=round(latency, 3))
    if "error" in d:
        row.update(error=d["error"], raw_output=None)
    else:
        ch = d["choices"][0]
        row.update(raw_output=ch["message"].get("content"), finish_reason=ch.get("finish_reason"),
                   provider=d.get("provider"), served_model=d.get("model"), usage=d.get("usage"))
        if j["condition"] == "C1L":
            lp = (ch.get("logprobs") or {}).get("content") or []
            row["top_logprobs"] = lp[0].get("top_logprobs") if lp else None
    with _lock:
        append_jsonl(out_path, row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pilot")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-items", type=int, default=None)
    ap.add_argument("--conditions", default=None, help="comma list, e.g. C1,C2")
    args = ap.parse_args()
    cfg = load_config(args.config)
    items = read_jsonl(ROOT / cfg["items"])
    if args.limit_items:
        # spread the smoke-test items across texts
        by_text = {}
        for it in items:
            by_text.setdefault(it["text_id"], []).append(it)
        picked = []
        while len(picked) < args.limit_items and any(by_text.values()):
            for k in list(by_text):
                if by_text[k] and len(picked) < args.limit_items:
                    picked.append(by_text[k].pop(0))
        items = picked
    if args.conditions:
        cfg["conditions"] = args.conditions.split(",")
    pricing = load_pricing(cfg)
    jobs = build_jobs(cfg, items)
    if args.dry_run:
        print(f"{len(items)} items, {len(cfg['models'])} models, {len(jobs)} model calls (+ judge)")
        print_estimate(estimate(cfg, jobs, items, pricing))
        return
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calls.jsonl"
    done = {job_key(r) for r in read_jsonl(out_path) if r.get("raw_output") is not None}
    todo = [j for j in jobs if job_key(j) not in done]
    print(f"{len(jobs)} jobs, {len(done)} already done, {len(todo)} to run")
    for phase in PHASE_ORDER:  # C1 phase completes before any C2 call is made
        batch = [j for j in todo if j["condition"] == phase]
        if not batch:
            continue
        n_err = 0
        with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as ex:
            futs = [ex.submit(run_job, cfg, pricing, j, out_path) for j in batch]
            for k, f in enumerate(as_completed(futs), 1):
                if "error" in f.result():
                    n_err += 1
                if k % 100 == 0 or k == len(batch):
                    print(f"  {phase}: {k}/{len(batch)} (errors {n_err})", flush=True)


if __name__ == "__main__":
    main()
