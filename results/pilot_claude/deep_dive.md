# Deep dive: pilot_claude

Every analysis here removes, or conditions away, the part of confidence that is shared item difficulty, and asks whether what is left carries model-specific knowledge. n = 120 items; treat single numbers as indicative.

## A. Private signal: is the model right when it disagrees with the outsiders about itself?

private = self C1 − mean of the two siblings' C3 about the same target. Both private signal and outcome are demeaned within text (so 'this text is easy' is removed), then correlated. Positive = when M is more optimistic than outsiders about itself, it tends to be right. Control row: the same statistic for each sibling's deviation from the other two (self + other sibling).

| target | signal | n | within-text r with outcome | 95% CI | AUROC of raw signal |
|---|---|---|---|---|---|
| opus-4-6 | self − outsiders | 120 | 0.004 | [-0.246, 0.275] | 0.242 |
| opus-4-6 | sonnet-4-6 − rest | 119 | -0.109 | [-0.343, 0.169] | 0.545 |
| opus-4-6 | haiku-4-5 − rest | 120 | 0.114 | [-0.151, 0.401] | 0.707 |
| sonnet-4-6 | self − outsiders | 120 | 0.046 | [-0.226, 0.312] | 0.471 |
| sonnet-4-6 | opus-4-6 − rest | 120 | -0.030 | [-0.302, 0.244] | 0.218 |
| sonnet-4-6 | haiku-4-5 − rest | 120 | -0.007 | [-0.266, 0.284] | 0.724 |
| haiku-4-5 | self − outsiders | 120 | -0.045 | [-0.135, 0.039] | 0.648 |
| haiku-4-5 | opus-4-6 − rest | 120 | -0.126 | [-0.206, -0.048] | 0.255 |
| haiku-4-5 | sonnet-4-6 − rest | 120 | 0.168 | [0.041, 0.298] | 0.486 |

## B. Relative self-knowledge on discordant items

Items where P and Q got different outcomes. P's view: C1_P − C3_{P→Q} (how much better P expects to do than Q). Q's view of the same contrast: C3_{Q→P} − C1_Q. AUROC for predicting 'P succeeded, Q failed'. 0.5 = no idea who wins.

| P | Q | discordant n (P won / Q won) | P's view AUROC | Q's view AUROC | P views itself better? |
|---|---|---|---|---|---|
| opus-4-6 | sonnet-4-6 | 13 (8/5) | 0.850 | 0.300 | yes |
| opus-4-6 | haiku-4-5 | 36 (36/0) | n/a | n/a | no |
| sonnet-4-6 | haiku-4-5 | 33 (33/0) | n/a | n/a | no |

## C. Projection: whose recall does an outsider's prediction track?

On items where predictor P and target Q have different outcomes, AUROC of P's prediction about Q against Q's outcome. Below 0.5 means P's prediction about Q follows P's *own* outcome: P assumes Q knows what P knows.

| predictor P | target Q | discordant n | AUROC vs Q's outcome | reading |
|---|---|---|---|---|
| opus-4-6 | sonnet-4-6 | 13 | 0.550 | no signal |
| opus-4-6 | haiku-4-5 | 36 | n/a | no signal |
| sonnet-4-6 | opus-4-6 | 13 | 0.388 | projects own knowledge |
| sonnet-4-6 | haiku-4-5 | 33 | n/a | no signal |
| haiku-4-5 | opus-4-6 | 36 | n/a | no signal |
| haiku-4-5 | sonnet-4-6 | 33 | n/a | no signal |

**Nested version** (knowledge turned out nested, so most pairs have one-sided discordance). Mean stated gist about the weaker model Q, split by outcome cell. If an outsider P projects, its numbers about Q stay high on items only P recalled.

| stronger P | weaker Q | predictor | both recalled | only P recalled | neither | drop (both → only P) |
|---|---|---|---|---|---|---|
| opus-4-6 | haiku-4-5 | opus-4-6 | 0.86 (56) | 0.75 (36) | 0.56 (28) | 0.11 |
| opus-4-6 | haiku-4-5 | sonnet-4-6 | 0.80 (56) | 0.64 (36) | 0.44 (28) | 0.16 |
| opus-4-6 | haiku-4-5 | haiku-4-5 (self) | 0.96 (56) | 0.79 (36) | 0.35 (28) | 0.18 |
| sonnet-4-6 | haiku-4-5 | opus-4-6 | 0.86 (56) | 0.76 (33) | 0.57 (31) | 0.09 |
| sonnet-4-6 | haiku-4-5 | sonnet-4-6 | 0.80 (56) | 0.67 (33) | 0.43 (31) | 0.13 |
| sonnet-4-6 | haiku-4-5 | haiku-4-5 (self) | 0.96 (56) | 0.79 (33) | 0.40 (31) | 0.18 |
| opus-4-6 | sonnet-4-6 | opus-4-6 | 0.88 (84) | 0.60 (8) | 0.58 (23) | 0.28 |
| opus-4-6 | sonnet-4-6 | sonnet-4-6 (self) | 0.88 (84) | 0.59 (8) | 0.50 (23) | 0.29 |
| opus-4-6 | sonnet-4-6 | haiku-4-5 | 0.79 (84) | 0.28 (8) | 0.34 (23) | 0.50 |

## D. Address errors: right text, wrong unit

A failed output (grade < 2) is an address error if it matches another unit of the same text with similarity ≥ 0.75.

| model_id   | text_id         |   failures |   address_errors |
|:-----------|:----------------|-----------:|-----------------:|
| haiku-4-5  | kjv             |          2 |                1 |
| haiku-4-5  | sonnets         |         14 |                3 |
| haiku-4-5  | tn_constitution |         15 |                0 |
| haiku-4-5  | tractatus       |         30 |                0 |
| haiku-4-5  | us_constitution |          3 |                0 |
| opus-4-6   | tn_constitution |         14 |                8 |
| opus-4-6   | tractatus       |         14 |                5 |
| sonnet-4-6 | tn_constitution |         15 |                5 |
| sonnet-4-6 | tractatus       |         16 |                0 |

Examples: opus-4-6 asked 2.2 → gave 2.02 (1.00); haiku-4-5 asked 118 → gave 60 (1.00); opus-4-6 asked Article XI, Section 16 → gave Article IX, Section 2 (1.00); sonnet-4-6 asked Article III, Section 14 → gave Article I, Section 8 (1.00); opus-4-6 asked Article I, Section 10 → gave Article I, Section 8 (1.00); opus-4-6 asked Article I, Section 27 → gave Article I, Section 26 (1.00); opus-4-6 asked Article IV, Section 2 → gave Article III, Section 2 (1.00); opus-4-6 asked Article XI, Section 2 → gave Article IX, Section 2 (1.00)

Does confidence track *content* (knows some unit of the text verbatim, address-error counted as known) or *address* (correct unit)?

| model | address errors | self AUROC vs correct unit | self AUROC vs content known | mean stated gist on address errors | on true failures |
|---|---|---|---|---|---|
| opus-4-6 | 13 | 0.929 | 0.936 | 0.72 | 0.59 |
| sonnet-4-6 | 5 | 0.925 | 0.874 | 0.36 | 0.56 |
| haiku-4-5 | 4 | 0.922 | 0.918 | 0.92 | 0.58 |

## E. Ratios: flagging failures and endorsing successes

Failure-flag rate = share of the target's failures that the predictor rated below 50%. Success-endorse rate = share of successes rated 50% or above. Implied P(verbatim | gist) = mean(verbatim)/mean(gist) of the predictor's numbers; observed = share of gist-correct attempts that were verbatim.

| target | predictor | failures flagged | successes endorsed | implied P(verb | gist) | observed P(verb | gist) |
|---|---|---|---|---|---|
| opus-4-6 | opus-4-6 (self) | 0.04 (28) | 0.99 | 0.82 | 0.90 |
| opus-4-6 | sonnet-4-6 | 0.71 (28) | 0.93 | 0.50 | 0.90 |
| opus-4-6 | haiku-4-5 | 0.89 (28) | 0.87 | 0.54 | 0.90 |
| sonnet-4-6 | opus-4-6 | 0.16 (31) | 0.99 | 0.69 | 0.87 |
| sonnet-4-6 | sonnet-4-6 (self) | 0.45 (31) | 0.98 | 0.73 | 0.87 |
| sonnet-4-6 | haiku-4-5 | 0.97 (31) | 0.84 | 0.57 | 0.87 |
| haiku-4-5 | opus-4-6 | 0.11 (64) | 0.98 | 0.65 | 0.88 |
| haiku-4-5 | sonnet-4-6 | 0.38 (64) | 0.95 | 0.46 | 0.88 |
| haiku-4-5 | haiku-4-5 (self) | 0.42 (64) | 1.00 | 0.80 | 0.88 |

## F. Continuous outcome: Spearman(stated gist, similarity)

Uses the full similarity score instead of the 0/1 grade. Within-text = mean of per-text Spearman (texts with variance).

| target | predictor | pooled ρ | within-text ρ |
|---|---|---|---|
| opus-4-6 | opus-4-6 (self) | 0.649 | 0.277 |
| opus-4-6 | sonnet-4-6 | 0.564 | 0.158 |
| opus-4-6 | haiku-4-5 | 0.691 | 0.151 |
| sonnet-4-6 | opus-4-6 | 0.615 | 0.175 |
| sonnet-4-6 | sonnet-4-6 (self) | 0.610 | 0.318 |
| sonnet-4-6 | haiku-4-5 | 0.663 | 0.247 |
| haiku-4-5 | opus-4-6 | 0.601 | 0.032 |
| haiku-4-5 | sonnet-4-6 | 0.566 | -0.021 |
| haiku-4-5 | haiku-4-5 (self) | 0.732 | 0.103 |

## G. Target-specific prediction: does a predictor know *which model* will get a given item?

For each item, take predictor P's three numbers (about Opus, Sonnet, Haiku) and subtract their mean; do the same for the three actual outcomes. Correlation across all item × target cells. Item difficulty cancels exactly; only 'who will get this one' remains. Also: share of P's prediction variance that is item-level (shared across targets).

| predictor | r(within-item prediction deviation, within-item outcome deviation) | 95% CI | item-level share of prediction variance |
|---|---|---|---|
| opus-4-6 | 0.310 | [0.205, 0.414] | 0.87 |
| sonnet-4-6 | 0.277 | [0.136, 0.413] | 0.91 |
| haiku-4-5 | -0.378 | [-0.488, -0.256] | 0.87 |

**Double-centered** (also removes each target's overall level, e.g. 'Haiku is weaker'): only item × target interaction remains.

| predictor | r(interaction) | 95% CI | mean stated gist: self vs about others |
|---|---|---|---|
| opus-4-6 | 0.084 | [-0.046, 0.222] | 0.85 vs 0.77 |
| sonnet-4-6 | 0.079 | [-0.087, 0.252] | 0.78 vs 0.68 |
| haiku-4-5 | -0.162 | [-0.293, -0.031] | 0.77 vs 0.65 |
