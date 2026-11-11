"""Prompt templates. Variant 'a' is the spec wording; 'b' is a paraphrase. Variant is assigned per item.

The spec's {item_noun} {locator} is extended to a `unit` description that also names the span
(e.g. "the first quatrain (lines 1-4) of Sonnet 127"), so C1, C2 and C3 all refer to exactly the
span that is scored.
"""

SYSTEM = {
    "a": ("You are participating in a research study on what language models remember about specific texts. "
          "Do not use tools or search. Answer from your own knowledge. Follow the output format exactly."),
    "b": ("This is a research study about what language models have memorized from particular texts. "
          "Answer only from your own memory, without tools or search, and follow the requested output format exactly."),
}


def unit_desc(it):
    base = f"{it['item_noun']} {it['locator']}".strip()
    span = it["span"]
    if span.startswith("first quatrain"):
        return f"the first quatrain (lines 1-4) of {base}"
    if span.startswith("first "):
        return f"the opening of {base} (its first {span.split()[1]} words)"
    return base


def source_desc(it):
    return f"{it['title']} by {it['author']} ({it['edition']})"


C1 = {
    "a": """You will later be asked to reproduce {unit} from {source}. Do not write it out now.

Estimate:
- exists: probability (0-100) that this item exists in that edition
- verbatim: probability (0-100) that your reproduction will be essentially word-for-word (at least 90% character match, all numbers and symbols correct)
- gist: probability (0-100) that your reproduction will state the same content, even if worded differently

Respond with only JSON: {{"exists": _, "verbatim": _, "gist": _}}""",
    "b": """In a later turn you will be asked to write out {unit} from {source} from memory. Do not write it now.

Give three probabilities from 0 to 100:
- exists: that this item is present in that edition
- verbatim: that what you write will be nearly word-for-word (90% or more of characters matching, every number and symbol right)
- gist: that what you write will convey the same content, even with different wording

Reply with only this JSON object: {{"exists": _, "verbatim": _, "gist": _}}""",
}

C2 = {
    "a": """Reproduce {unit} from {source} as exactly as you can. {span_instruction} Use LaTeX for mathematics. Output only the text of the item, with no commentary.

Always give your best attempt, even if unsure. If you are confident the item does not exist, output only: NONEXISTENT""",
    "b": """Write out {unit} from {source}, word for word as far as you can recall it. {span_instruction} Use LaTeX for any mathematics. Give only the text itself, with no comments or preamble.

Make your best attempt even if you are unsure. Only if you are confident that no such item exists, output just: NONEXISTENT""",
}

C3 = {
    "a": """Another AI model, {target_name}, will be asked to reproduce {unit} from {source} from memory, with no tools.

Estimate the probability (0-100) that its reproduction will be: verbatim (at least 90% character match, all numbers and symbols correct), and gist-correct. Also estimate the probability the item exists.

Respond with only JSON: {{"exists": _, "verbatim": _, "gist": _}}""",
    "b": """An AI model called {target_name} will be asked to write out {unit} from {source} from memory, without tools.

Give probabilities from 0 to 100 that what it writes will be verbatim (90% or more of characters matching, every number and symbol right) and that it will convey the correct content (gist). Also give the probability that the item exists.

Reply with only this JSON object: {{"exists": _, "verbatim": _, "gist": _}}""",
}

# Secondary confidence measure for models whose verbal numbers are near-constant:
# P("Yes") from the first-token logprobs. One output token.
C1L = {
    "a": """Could you correctly reproduce the content of {unit} from {source} from memory (exact wording not required)? Answer with one word: Yes or No.""",
    "b": """From memory alone, can you state what {unit} of {source} says (the content, not necessarily the exact words)? Reply with a single word: Yes or No.""",
}

JUDGE_SYSTEM = "You grade memory-recall attempts against a reference text. Follow the rubric exactly and reply with JSON only."

JUDGE = """Reference text ({unit} from {source}):
<reference>
{ground_truth}
</reference>

Attempt by a model asked to reproduce it from memory:
<attempt>
{output}
</attempt>

Rubric:
3 Verbatim: essentially word-for-word (at least 90% character match) and every number, name and key term correct.
2 Gist: the same content and claims, worded differently, or verbatim with a minor error in a number, name or key term.
1 Topic: right subject and kind of passage, wrong specifics (for example a different verse or proposition on the same theme).
0 Wrong: different topic, a refusal, empty, or "NONEXISTENT".

Ignore differences in capitalization, punctuation, archaic vs modern spelling, and line breaks. Grade only the span shown in the reference; ignore extra text after it.

Reply with only JSON: {{"grade": 0|1|2|3, "reason": "<at most 15 words>"}}"""


def build(cond, it, variant, target_name=None, output=None):
    kw = dict(unit=unit_desc(it), source=source_desc(it), span_instruction=it["span_instruction"])
    if cond == "C1":
        return SYSTEM[variant], C1[variant].format(**kw)
    if cond == "C2":
        return SYSTEM[variant], C2[variant].format(**kw)
    if cond in ("C3", "C3self"):
        return SYSTEM[variant], C3[variant].format(target_name=target_name, **kw)
    if cond == "C1L":
        return SYSTEM[variant], C1L[variant].format(**kw)
    if cond == "JUDGE":
        return JUDGE_SYSTEM, JUDGE.format(ground_truth=it["ground_truth"], output=output, **kw)
    raise ValueError(cond)
