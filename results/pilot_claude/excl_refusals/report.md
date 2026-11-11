# Pilot report: pilot_claude (outcome: gist, grade ≥ 2)

120 items · models: opus-4-6, sonnet-4-6, haiku-4-5 · judge: claude-haiku-4-5 · backend: claude_cli

Graded recall attempts per model: opus-4-6 120, sonnet-4-6 120, haiku-4-5 74

## 1. Data quality

| model | base rate | grades 0/1/2/3 | refusals | C1 parse fail | C3 parse fail | C1 leaks | C1 thinking tok (max) | C1 gist SD | C1 distinct |
|---|---|---|---|---|---|---|---|---|---|
| opus-4-6 | 0.77 | 25/3/9/83 | 0 | 0.00 | 0.00 | 0 | 0 | 16.8 | 19 |
| sonnet-4-6 | 0.74 | 29/2/12/77 | 0 | 0.00 | 0.00 | 0 | 0 | 20.5 | 20 |
| haiku-4-5 | 0.76 | 10/8/7/49 | 46 | 0.00 | 0.00 | 0 | 0 | 31.3 | 23 |

Recall base rate by text/tier:

| stratum                 |   opus-4-6 |   sonnet-4-6 |   haiku-4-5 |
|:------------------------|-----------:|-------------:|------------:|
| kjv/famous              |       1    |          1   |        1    |
| kjv/random              |       1    |          1   |        0.9  |
| sonnets/famous          |       1    |          1   |        0.8  |
| sonnets/random          |       1    |          1   |        0.4  |
| tn_constitution/obscure |       0.07 |          0   |      nan    |
| tractatus/deep          |       0.5  |          0.4 |      nan    |
| tractatus/shallow       |       0.6  |          0.6 |      nan    |
| us_constitution/famous  |       1    |          1   |        0.86 |

Refusals by text: tn_constitution/haiku-4-5: 15; tractatus/haiku-4-5: 30; us_constitution/haiku-4-5: 1

## 2. AUROC matrix (rows predict columns; diagonal = self C1, off-diagonal = C3)

**gist** (outcome grade ≥ 2, predictor field `gist`)

|            |   opus-4-6 |   sonnet-4-6 |   haiku-4-5 |
|:-----------|-----------:|-------------:|------------:|
| opus-4-6   |      0.929 |        0.927 |       0.621 |
| sonnet-4-6 |      0.896 |        0.925 |       0.697 |
| haiku-4-5  |      0.929 |        0.926 |       0.792 |

**verbatim** (outcome grade ≥ 3, predictor field `verbatim`)

|            |   opus-4-6 |   sonnet-4-6 |   haiku-4-5 |
|:-----------|-----------:|-------------:|------------:|
| opus-4-6   |      0.938 |        0.886 |       0.595 |
| sonnet-4-6 |      0.882 |        0.848 |       0.653 |
| haiku-4-5  |      0.916 |        0.884 |       0.695 |

## 3. Self-advantage (H1)

Δ = AUROC(self C1 → M) − AUROC(comparator → M), same items, paired bootstrap (2,000 resamples). Comparators: each sibling's C3 prediction, and M's own prediction under the C3 'another model' framing.

| target | comparator | self AUROC | comparator AUROC | Δ | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|---|---|---|
| opus-4-6 | sonnet-4-6 (C3) [matched] | 0.929 | 0.896 | 0.033 | [-0.027, 0.101] | 0.137 |
| opus-4-6 | haiku-4-5 (C3) | 0.929 | 0.929 | 0.000 | [-0.041, 0.040] | 0.486 |
| opus-4-6 | self, C3 framing | 0.929 | 0.909 | 0.020 | [-0.014, 0.055] | 0.121 |
| sonnet-4-6 | opus-4-6 (C3) [matched] | 0.925 | 0.927 | -0.002 | [-0.048, 0.042] | 0.555 |
| sonnet-4-6 | haiku-4-5 (C3) | 0.925 | 0.926 | -0.001 | [-0.055, 0.051] | 0.516 |
| sonnet-4-6 | self, C3 framing | 0.925 | 0.904 | 0.021 | [-0.024, 0.078] | 0.214 |
| haiku-4-5 | opus-4-6 (C3) | 0.792 | 0.621 | 0.171 | [0.051, 0.283] | 0.003 |
| haiku-4-5 | sonnet-4-6 (C3) [matched] | 0.792 | 0.697 | 0.094 | [-0.024, 0.202] | 0.063 |
| haiku-4-5 | self, C3 framing | 0.792 | 0.717 | 0.074 | [-0.037, 0.190] | 0.104 |

## 4. Within-text AUROC (removes between-text popularity)

AUROC computed inside each text (n-weighted mean over texts where the target has both outcomes). A predictor that only knows 'KJV is easy, Tennessee is hard' scores ~0.5 here.

| target | predictor | pooled AUROC | within-text AUROC | Δ within (self − predictor) | 95% CI (stratified bootstrap) |
|---|---|---|---|---|---|
| opus-4-6 | self (opus-4-6) | 0.929 | 0.624 | | |
| opus-4-6 | sonnet-4-6 (C3) | 0.896 | 0.444 | 0.180 | [-0.142, 0.356] |
| opus-4-6 | haiku-4-5 (C3) | 0.929 | 0.466 | 0.158 | [-0.145, 0.304] |
| opus-4-6 | log n-gram freq | 0.801 | 0.509 | 0.115 | [-0.106, 0.375] |
| sonnet-4-6 | self (sonnet-4-6) | 0.925 | 0.721 | | |
| sonnet-4-6 | opus-4-6 (C3) | 0.927 | 0.667 | 0.054 | [-0.127, 0.244] |
| sonnet-4-6 | haiku-4-5 (C3) | 0.926 | 0.589 | 0.132 | [-0.043, 0.314] |
| sonnet-4-6 | log n-gram freq | 0.823 | 0.629 | 0.092 | [-0.179, 0.352] |
| haiku-4-5 | self (haiku-4-5) | 0.792 | 0.709 | | |
| haiku-4-5 | opus-4-6 (C3) | 0.621 | 0.569 | 0.141 | [-0.037, 0.295] |
| haiku-4-5 | sonnet-4-6 (C3) | 0.697 | 0.635 | 0.074 | [-0.149, 0.281] |
| haiku-4-5 | log n-gram freq | 0.792 | 0.583 | 0.127 | [-0.079, 0.306] |

## 5. Incremental information: does self-confidence predict M's recall beyond what outsiders know?

Logistic regression of M's outcome on text fixed effects + log n-gram frequency + both siblings' C3 predictions (base), then adding M's own C1 prediction (full). Likelihood-ratio test, 1 df. Reverse test: does each sibling add beyond self + controls? Predictors standardized; small ridge (λ=0.01) for stability.

| target | n | LR χ² (self added) | p | self coef | LR χ² (best sibling added to self+controls) | p |
|---|---|---|---|---|---|---|
| opus-4-6 | 119 | 0.83 | 0.3629 | 0.40 | 0.82 (haiku-4-5) | 0.3648 |
| sonnet-4-6 | 120 | 0.72 | 0.3971 | 0.61 | 0.16 (opus-4-6) | 0.6927 |
| haiku-4-5 | 74 | 1.36 | 0.2443 | 0.60 | 8.42 (opus-4-6) | 0.0037 |

## 6. Outsider baselines (not in the spec)

| target | self C1 | popularity tier | log n-gram freq | siblings' mean actual grade | best sibling's actual grade |
|---|---|---|---|---|---|
| opus-4-6 | 0.929 | 0.651 | 0.801 | 0.891 | 0.916 |
| sonnet-4-6 | 0.925 | 0.666 | 0.823 | 0.877 | 0.916 |
| haiku-4-5 | 0.792 | 0.657 | 0.792 | 0.464 | 0.491 |

## 7. Calibration of stated P(gist)

| target | predictor | mean stated | observed | overconfidence | Brier | ECE (5 bins) |
|---|---|---|---|---|---|---|
| opus-4-6 | opus-4-6 (self) | 0.85 | 0.77 | +0.09 | 0.120 | 0.104 |
| opus-4-6 | sonnet-4-6 | 0.70 | 0.76 | -0.07 | 0.116 | 0.121 |
| opus-4-6 | haiku-4-5 | 0.66 | 0.77 | -0.11 | 0.113 | 0.125 |
| sonnet-4-6 | opus-4-6 | 0.79 | 0.74 | +0.05 | 0.119 | 0.132 |
| sonnet-4-6 | sonnet-4-6 (self) | 0.78 | 0.74 | +0.03 | 0.105 | 0.083 |
| sonnet-4-6 | haiku-4-5 | 0.65 | 0.74 | -0.09 | 0.112 | 0.152 |
| haiku-4-5 | opus-4-6 | 0.85 | 0.76 | +0.09 | 0.202 | 0.146 |
| haiku-4-5 | sonnet-4-6 | 0.77 | 0.76 | +0.02 | 0.172 | 0.084 |
| haiku-4-5 | haiku-4-5 (self) | 0.95 | 0.76 | +0.20 | 0.210 | 0.197 |

## 8. Framing control, prompt sensitivity, disagreements, scoring checks

**Framing control** (same model, same items: C1 'you' vs C3 'another AI model, {own name}')

| model | AUROC C1 | AUROC C3-self | mean stated gist C1 | mean stated gist C3-self |
|---|---|---|---|---|
| opus-4-6 | 0.929 | 0.909 | 85.4 | 80.5 |
| sonnet-4-6 | 0.925 | 0.904 | 77.7 | 66.0 |
| haiku-4-5 | 0.792 | 0.717 | 76.8 | 62.2 |

**Prompt-variant sensitivity** (self C1 AUROC and base rate by template paraphrase)

| model | AUROC variant a | AUROC variant b | base rate a | base rate b | n a / n b |
|---|---|---|---|---|---|
| opus-4-6 | 0.928 | 0.929 | 0.70 | 0.83 | 61 / 59 |
| sonnet-4-6 | 0.903 | 0.946 | 0.69 | 0.80 | 61 / 59 |
| haiku-4-5 | 0.900 | 0.677 | 0.74 | 0.78 | 61 / 59 |

**Largest self-vs-matched disagreements** (stated gist; outcome = target's actual grade)

| target | item | locator | self | matched | grade | who was closer |
|---|---|---|---|---|---|---|
| opus-4-6 | tractatus-2.0211 | 2.0211 | 25 | 85 | 3 | sonnet-4-6 |
| opus-4-6 | tnconst-1-27 | Article I, Section 27 | 80 | 25 | 0 | sonnet-4-6 |
| opus-4-6 | tnconst-1-22 | Article I, Section 22 | 82 | 28 | 0 | sonnet-4-6 |
| opus-4-6 | tractatus-3.143 | 3.143 | 60 | 8 | 3 | self |
| sonnet-4-6 | tractatus-5.473 | 5.473 | 75 | 30 | 2 | self |
| sonnet-4-6 | tractatus-4.461 | 4.461 | 80 | 45 | 1 | opus-4-6 |
| sonnet-4-6 | tnconst-11-16 | Article XI, Section 16 | 20 | 55 | 0 | self |
| sonnet-4-6 | tractatus-4.0311 | 4.0311 | 88 | 55 | 2 | self |
| haiku-4-5 | kjv-Jeremiah-41-4 | Jeremiah 41:4 | 95 | 40 | 3 | self |
| haiku-4-5 | kjv-2Kings-12-13 | 2 Kings 12:13 | 92 | 45 | 3 | self |
| haiku-4-5 | sonnets-095 | 95 | 88 | 45 | 0 | sonnet-4-6 |
| haiku-4-5 | kjv-Leviticus-5-4 | Leviticus 5:4 | 98 | 55 | 3 | self |

**Scoring checks**

- Rule-graded 205 of 360; judged 155; ungraded 46.
- Spec shortcut (sim < 0.3 → 0, sim ≥ 0.9 → 3/2) agrees with the judge on 91% of 96 overlapping outputs; judge gave ≥ 2 to 2 outputs the shortcut would have scored 0.
- Similarity quartiles by model: haiku-4-5 0.25/0.62/1.0; opus-4-6 0.41/1.0/1.0; sonnet-4-6 0.32/0.99/1.0
- Judge–human agreement (kappa) not yet measured: needs a human-graded stratified sample.

## 9. Kill criteria and redesign triggers

- opus-4-6: ok
- sonnet-4-6: ok
- haiku-4-5: ok

Figures: `figures/fig0_summary.png`, `fig1_base_rates.png`, `fig2_auroc_matrix.png`, `fig3_delta_forest.png`, `fig4_within_text.png`, `fig5_calibration.png`.