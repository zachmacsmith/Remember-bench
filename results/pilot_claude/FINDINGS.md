# Pilot findings: Claude-only, thinking off (2026-11-11)

Run: 120 public-domain items, Claude Opus 4.6, Sonnet 4.6 and Haiku 4.5 via the Claude Code CLI. 1,800 calls (C1, C3, C3-self, C2) plus 155 judge calls. No errors, 0 thinking tokens, 2 parse failures out of 1,440 predictions. $9.38 at list price. Full tables in `report.md`, figures in `figures/`.

## Result in one paragraph

Self-prediction beat sibling prediction clearly for **one of three targets (Haiku 4.5: Δ ≈ +0.10 AUROC, 95% CI excludes 0)** and not at all for Opus 4.6 or Sonnet 4.6 (Δ ≈ 0). The pooled AUROC (~0.93 for nearly every predictor) is dominated by between-text popularity, because Opus and Sonnet recall 100% of KJV, Sonnets and US Constitution items. Within texts, self beat both siblings in all 6 comparisons (Δ +0.05 to +0.18), but every confidence interval crosses 0. The direction is consistent; the pilot is underpowered exactly where the question lives.

## What the data say

1. **Range restriction is the dominant problem.** Opus and Sonnet are at ceiling in 5 of 8 strata. All their outcome variance is in Tractatus (40–60% recall) and the Tennessee Constitution (0–7%). Any predictor that knows "Tennessee is obscure" gets AUROC ~0.9, so the pooled H1 test mostly measures popularity estimation.
2. **The Haiku self-advantage looks like outsiders projecting, not only insider access.** Opus and Sonnet predict Haiku's recall as if Haiku knew roughly what they know: stated 0.76 and 0.67 vs. actual 0.47, and their ranking is worse (0.81–0.82 vs. self 0.92). Haiku's own self-report survives the controls:
   - It beats siblings' *actual* recall of the same items (0.82).
   - It beats n-gram frequency (0.87).
   - It still predicts its own recall under third-person framing (C3-self 0.89).
   - It adds information beyond text, frequency and both siblings' predictions (logistic LR test p = 0.045).

   Opus's prediction about Haiku also adds independent information beyond Haiku's self-report (p = 0.008). Neither source dominates.
3. **For Opus and Sonnet, outsiders' actual recall predicts better than self-report.** The siblings' mean actual grade on an item predicts Opus's recall at 0.944 and Sonnet's at 0.952, against self 0.929 and 0.925. "Did the other model get it?" is a better signal than "do I think I'll get it?" when the models share most of their knowledge.
4. **Self-framing changes confidence more than it changes discrimination.** Asked about themselves in the third person (C3-self), all three models state 5–15 points lower confidence. Discrimination drops only slightly (Δ +0.02 to +0.03, none significant). The self-advantage that exists is not a framing artifact.
5. **Calibration and discrimination diverge.** Haiku has the best discrimination about itself and the worst calibration: it is overconfident by 30 points (stated 0.77, observed 0.47). Opus is overconfident about itself (+0.09); Sonnet is nearly calibrated (+0.03).
6. **Address errors drive Opus's worst misses.** Opus stated ~80% for Tennessee Art. I §22 and §27 and failed both. In both cases it reproduced **§26 (right to bear arms) verbatim**. It knows the text but not the locator-to-text map. Recall of addressable units has two parts, content memory and address memory, and the current grade conflates them. Opus's confidence appears to track content familiarity.
7. **The scoring shortcut is mostly safe.** The spec's rule band (sim < 0.3 → 0, sim ≥ 0.9 → verbatim) agreed with the judge on 91% of outputs both graded. Only 2 outputs below 0.3 were judged gist-correct. Judge–human agreement is still unmeasured.
8. **The pilot passed all kill criteria.** Parse failures were 0.1%. Confidence varied (SD 17–31, 19–23 distinct values). Base rates were 0.47–0.77. Haiku refused 3 items (all Tennessee). No C1 leaks after excluding Opus 5.5.

## Caveats

- One family. Siblings share training data, so this is the spec's shared-data control, not a cross-family test.
- In-family judge (Haiku 4.5). Labels are shared by all predictors of a target, so Δ is not biased, but base rates may be.
- No temperature control through the CLI.
- 120 items. Within-text tests use only the strata with outcome variance (about 45 items for Opus and Sonnet).
- The prompt-variant comparison is confounded with item assignment (variant is randomized per item), so its AUROC gap (Haiku 0.97 vs. 0.86) is not interpretable as prompt sensitivity.

## What to change for the next run

1. **Rebalance items to the informative band.** Drop most famous KJV, Sonnets and US Constitution items. Add deep Tractatus, Tennessee, Spinoza, Beezer, obscure RFC sections, and later sonnet lines (beyond the first quatrain). Target 40–60% recall per model per text, and 300+ items in that band. The within-text test is the real test, and it needs the items.
2. **Add an informed-outsider baseline.** Give the C3 predictor the target's overall base rate, or 10 example outcomes. If self still wins, the advantage is item-level self-knowledge, not knowledge of one's own general ability.
3. **Score address memory separately.** Tag outputs that match a *different* unit of the same text (as with Opus's §26) as "address error". This is a cheap automatic check against the full text.
4. **Add cross-family models** (the open ladder via OpenRouter) for a true outsider comparison and H2.
5. **Add nonexistent locators** (about 10%) so `exists` and `NONEXISTENT` can be evaluated.
6. **Human-grade a stratified sample** of about 150 items for judge kappa.
