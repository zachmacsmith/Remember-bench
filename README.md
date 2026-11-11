# remember-bench: verbatim recall introspection

Tests whether a model predicts its own recall of addressable text units ("Sonnet 127, first quatrain")
better than other models can predict it (H1), and whether that self-advantage grows with size (H2).
The full spec is not reproduced here. This repo implements a **minimal pilot**.

## Pilot design (the smallest run that can fail informatively)

| | Pilot | Full spec |
|---|---|---|
| Items | 120: KJV, Sonnets, Tractatus, US + TN Constitutions (30 each) | 640 over 12 texts |
| Models | Llama 3.1 8B, Llama 3.3 70B, Gemma 3 4B, Gemma 3 27B | OLMo + Qwen ladders + 3 frontier |
| Conditions | C1, C2, C3 (full 4×4 matrix), C3-self framing, C1L (P(Yes) from logprobs) | C1, C2, C3 |
| Calls | 3,360 model calls + ~480 judge calls | ~92,000 |
| Cost | **≈ $0.20** at OpenRouter prices on 2026-11-11 | ≈ $34 (11-model projection, thinking off) |
| Wall time | under an hour at concurrency 8 | |

Design choices:
- **Public-domain prose only.** Ground truth is extracted programmatically, with no transcription. No copyright refusals, so refusal handling and edition questions are not confounds yet. Cost: the pilot says nothing about math texts, where reconstruction and memory diverge most.
- **Built-in popularity gradient in every text.** Famous vs. random verses and sonnets, shallow (≤1 decimal) vs. deep (≥3 decimals) Tractatus propositions, US vs. TN constitution. Without this spread, AUROC has nothing to discriminate.
- **Models swapped from the spec.** OLMo is not served on OpenRouter; testing it means self-hosting on a GPU. Llama 3.x and Gemma 3 are two dense families with public sizes and cents-level prices. Pairs: Llama 8B ↔ Gemma 4B and Llama 70B ↔ Gemma 27B. Matching is loose: 70B vs. 27B is not capability-matched.
- **Judge:** Claude Haiku 5.5, outside both families. The pilot judges every output the rule can't grade, not only the spec's 0.3–0.9 similarity band. The report checks the spec's shortcut against the judge.

What the pilot can and cannot tell you:
- **Can:** whether C1 confidence varies at all, parse-failure rates, base rates per model and text, grade distribution, and whether a large Δ exists (roughly Δ ≥ 0.08; with 120 items a single AUROC has SE ≈ 0.05).
- **Cannot:** detect the spec's 0.03–0.05 effects, test H2 (two sizes per family is a sign, not a trend), or estimate refusal rates on copyrighted texts.

## Run

```bash
python -I src/build_items.py                 # data/items/pilot_items.csv + pilot_ground_truth.jsonl (committed)
python -I src/popularity.py                  # optional: infini-gram log-frequency covariate (~360 queries, free)
python -I src/run_calls.py --dry-run         # call counts and cost estimate
python -I src/run_calls.py --limit-items 8   # smoke test (~$0.02)
python -I src/run_calls.py                   # full pilot; resumable
python -I src/score.py                       # parse predictions, score recall, run judge
python -I src/analyze.py                     # results/pilot/report.md
```

Needs `OPENROUTER_API_KEY`. In the cloud environment the network proxy injects it.

## Layout

- `data/raw/`: downloaded texts (`data/SOURCES.md` has editions, licenses, caveats; `scripts/fetch_texts.sh` re-fetches)
- `data/items/`: pilot item set (public schema + ground truth; all public domain)
- `configs/pilot.json`: models, matched pairs, token caps, judge
- `src/prompts.py`: spec templates (variant a) plus a paraphrase (variant b), assigned per item

## Known weaknesses to fix before the full run

1. **The outsider baseline in the spec is weak.** A C3 predictor guesses from item popularity. If self-prediction beats it, that may only mean the self-predictor is better at estimating difficulty in general. `analyze.py` adds stronger baselines: popularity tier, n-gram frequency, and other models' *actual* recall of the item. Self-report only demonstrates privileged information if it beats those.
2. **Matched-pair capability is not established.** The pilot has no benchmark matching. The full run should match on a recall benchmark (e.g. base rate on a held-out item set), not just parameter count.
3. **Span specification.** Prose units are capped at 60 words and scored by prefix alignment. The prompt names the span ("its first 60 words") so C1, C2 and C3 refer to the same target.
4. **US Constitution text** is Gutenberg #5, not the National Archives transcription. TN text was supplied by hand and needs verification.
5. **Critical elements** are a prose heuristic (numbers, number words, capitalized non-initial words). Math items need formula and variable extraction.
