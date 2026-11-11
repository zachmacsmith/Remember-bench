"""Build the pilot item set from pinned public-domain sources.

Pilot texts (all public domain, programmatic ground truth, no copyright-refusal confound):
  kjv        King James Bible (Project Gutenberg #10; 1769 Oxford text)
  sonnets    Shakespeare, Sonnets (Project Gutenberg #1041), first quatrain
  tractatus  Wittgenstein, Tractatus, Ogden 1922 translation (Project Gutenberg #5740, TeX source)
  legal      US Constitution (Project Gutenberg #5) + Tennessee Constitution (user-supplied text)

Usage: python -I src/build_items.py [--per-text 30] [--seed 7]
Writes data/items/pilot_items.csv (public schema) and data/items/pilot_ground_truth.jsonl.
"""
import argparse
import csv
import json
import random
import re
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import RAW, ITEMS, sha256, clean_quotes, collapse_ws, first_n_words, word_count  # noqa: E402

MAX_WORDS = 60  # fixed span cap for prose units so exact match is comparable across texts

# ---------------------------------------------------------------- KJV
KJV_SHORT = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
    "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
    "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon",
    "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah",
    "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians",
    "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
    "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
]
KJV_FAMOUS = [
    ("John", 3, 16), ("Genesis", 1, 1), ("Romans", 8, 28), ("Jeremiah", 29, 11),
    ("Proverbs", 3, 5), ("Isaiah", 40, 31), ("Psalms", 23, 4), ("Joshua", 1, 9),
    ("Matthew", 6, 33), ("1 Corinthians", 13, 4),
]


def parse_kjv():
    txt = (RAW / "kjv.txt").read_text(encoding="utf-8-sig").replace("\r", "")
    body = txt.split("*** START OF THE PROJECT GUTENBERG EBOOK")[1].split("*** END OF THE PROJECT GUTENBERG EBOOK")[0]
    lines = body.split("\n")[1:]
    # table of contents: first 66 non-empty lines that are not testament headers
    toc = []
    i = 0
    while len(toc) < 66:
        ln = lines[i].strip()
        if ln and not ln.startswith("The Old Testament") and not ln.startswith("The New Testament"):
            toc.append(ln)
        i += 1
    rest = "\n".join(lines[i:])
    # locate each book title on its own line, in order
    positions = []
    cursor = 0
    for title in toc:
        m = re.compile(r"^" + re.escape(title) + r"\s*$", re.M).search(rest, cursor)
        if m is None:
            raise RuntimeError(f"KJV: book title not found: {title}")
        positions.append((m.start(), m.end()))
        cursor = m.end()
    verses = []  # (book, ch, vs, text)
    for bi, title in enumerate(toc):
        start = positions[bi][1]
        end = positions[bi + 1][0] if bi + 1 < len(toc) else len(rest)
        chunk = rest[start:end]
        if bi == 38:  # Malachi: strip the New Testament header that follows
            chunk = chunk.split("The New Testament of the King James Bible")[0]
        parts = re.split(r"(\d+):(\d+)\s", chunk)
        for j in range(1, len(parts) - 2, 3):
            ch, vs, t = int(parts[j]), int(parts[j + 1]), parts[j + 2]
            verses.append((KJV_SHORT[bi], ch, vs, collapse_ws(clean_quotes(t)).rstrip(" *")))
    return verses


def kjv_items(n, rng):
    verses = parse_kjv()
    idx = {(b, c, v): k for k, (b, c, v, _) in enumerate(verses)}
    total = len(verses)
    items = []
    for b, c, v in KJV_FAMOUS:
        k = idx[(b, c, v)]
        items.append(make_kjv(verses[k], k, total, "famous"))
    n_rand = n - len(items)
    famous_keys = {idx[f] for f in KJV_FAMOUS}
    for terc in range(3):
        lo, hi = terc * total // 3, (terc + 1) * total // 3
        pool = [k for k in range(lo, hi) if k not in famous_keys and 15 <= word_count(verses[k][3]) <= MAX_WORDS]
        take = n_rand // 3 + (1 if terc < n_rand % 3 else 0)
        for k in rng.sample(pool, take):
            items.append(make_kjv(verses[k], k, total, "random"))
    return items


def make_kjv(verse, k, total, tier):
    b, c, v, t = verse
    return dict(
        text_id="kjv", item_id=f"kjv-{b.replace(' ', '')}-{c}-{v}",
        title="the Holy Bible", author="the King James Version translators", edition="King James Version, 1769 Oxford standard text",
        item_noun="verse", locator=f"{b} {c}:{v}", span="full verse",
        span_instruction="Reproduce the full verse.",
        item_type="numbered_prose_unit", position_tercile=1 + 3 * k // total, popularity_tier=tier,
        ground_truth=t,
    )


# ---------------------------------------------------------------- Sonnets
SONNETS_FAMOUS = [18, 116, 130, 73, 29, 1, 55, 30, 60, 138]


def roman_to_int(s):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    tot, prev = 0, 0
    for ch in reversed(s):
        v = vals[ch]
        tot = tot - v if v < prev else tot + v
        prev = max(prev, v)
    return tot


def parse_sonnets():
    txt = (RAW / "shakespeare_sonnets.txt").read_text(encoding="utf-8-sig").replace("\r", "")
    body = txt.split("*** START OF THE PROJECT GUTENBERG EBOOK")[1].split("*** END OF THE PROJECT GUTENBERG EBOOK")[0]
    body = body.split("THE SONNETS", 1)[1]
    blocks = re.split(r"\n\s*([IVXLC]+)\s*\n", body)
    sonnets = {}
    for j in range(1, len(blocks) - 1, 2):
        num = roman_to_int(blocks[j])
        lines = [clean_quotes(ln.strip()) for ln in blocks[j + 1].split("\n") if ln.strip()]
        if lines and num not in sonnets:
            sonnets[num] = lines
    if len(sonnets) != 154:
        raise RuntimeError(f"Sonnets: parsed {len(sonnets)} sonnets, expected 154")
    return sonnets


def sonnet_items(n, rng):
    s = parse_sonnets()
    items = [make_sonnet(k, s[k], "famous") for k in SONNETS_FAMOUS]
    n_rand = n - len(items)
    for terc in range(3):
        lo, hi = 1 + terc * 154 // 3, 1 + (terc + 1) * 154 // 3
        pool = [k for k in range(lo, hi) if k not in SONNETS_FAMOUS]
        take = n_rand // 3 + (1 if terc < n_rand % 3 else 0)
        for k in rng.sample(pool, take):
            items.append(make_sonnet(k, s[k], "random"))
    return items


def make_sonnet(num, lines, tier):
    return dict(
        text_id="sonnets", item_id=f"sonnets-{num:03d}",
        title="Shake-speares Sonnets", author="William Shakespeare", edition="1609 quarto text, modern-spelling Project Gutenberg edition",
        item_noun="Sonnet", locator=f"{num}", span="first quatrain (lines 1-4)",
        span_instruction="Reproduce only the first quatrain (lines 1-4), one line per line.",
        item_type="numbered_verse_unit", position_tercile=1 + 3 * (num - 1) // 154, popularity_tier=tier,
        ground_truth="\n".join(lines[:4]),
    )


# ---------------------------------------------------------------- Tractatus
def brace_arg(s, i):
    """Return (content, end_index) of the {...} group starting at s[i] == '{'."""
    assert s[i] == "{"
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def strip_tex(t):
    t = re.sub(r"(?m)^%.*$", "", t)
    t = re.sub(r"(?<!\\)%.*$", "", t, flags=re.M)
    # remove footnotes entirely
    while "\\footnote{" in t:
        i = t.index("\\footnote{")
        _, e = brace_arg(t, i + len("\\footnote"))
        t = t[:i] + t[e:]
    for _ in range(3):
        t = re.sub(r"\\(?:emph|textit|textbf|textsc|mbox|textrm)\{([^{}]*)\}", r"\1", t)
    t = t.replace("\\;", " ").replace("\\,", " ").replace("~", " ").replace("\\ ", " ")
    t = t.replace("``", '"').replace("''", '"').replace("---", "--")
    return collapse_ws(clean_quotes(t))


def parse_tractatus():
    tex = (RAW / "tractatus_ogden.tex").read_text(encoding="utf-8", errors="replace")
    out = {}
    for m in re.finditer(r"\\PropositionE\{([0-9.]+)\}\s*", tex):
        num = m.group(1)
        j = m.end()
        if j >= len(tex) or tex[j] != "{":
            continue
        raw, _ = brace_arg(tex, j)
        if num in out:
            continue
        mathy = bool(re.search(r"\$|\\\(|\\\[|\\begin|\\Illustration|\\mathrm|\\sim|\\supset|\\cdot", raw))
        out[num] = (strip_tex(raw), mathy)
    if len(out) < 500:
        raise RuntimeError(f"Tractatus: parsed only {len(out)} propositions")
    return out


def tractatus_items(n, rng):
    props = parse_tractatus()
    nums = list(props)
    order = {k: i for i, k in enumerate(nums)}

    def depth(num):
        return len(num.split(".")[1]) if "." in num else 0

    shallow = [k for k in nums if depth(k) <= 1 and not props[k][1] and word_count(props[k][0]) >= 6]
    deep = [k for k in nums if depth(k) >= 3 and not props[k][1] and word_count(props[k][0]) >= 12]
    n_sh = min(10, len(shallow))
    items = [make_tract(k, props[k][0], order[k], len(nums), "shallow") for k in rng.sample(shallow, n_sh)]
    n_rand = n - n_sh
    total = len(nums)
    for terc in range(3):
        pool = [k for k in deep if terc * total // 3 <= order[k] < (terc + 1) * total // 3]
        take = n_rand // 3 + (1 if terc < n_rand % 3 else 0)
        for k in rng.sample(pool, take):
            items.append(make_tract(k, props[k][0], order[k], total, "deep"))
    return items


def make_tract(num, text, pos, total, tier):
    wc = word_count(text)
    truncated = wc > MAX_WORDS
    gt = first_n_words(text, MAX_WORDS) if truncated else text
    return dict(
        text_id="tractatus", item_id=f"tractatus-{num}",
        title="Tractatus Logico-Philosophicus", author="Ludwig Wittgenstein", edition="C. K. Ogden translation, 1922",
        item_noun="Proposition", locator=num,
        span=f"first {MAX_WORDS} words" if truncated else "full proposition",
        span_instruction=(f"Reproduce only the opening of the proposition (about the first {MAX_WORDS} words)."
                          if truncated else "Reproduce the full proposition."),
        item_type="numbered_prose_unit", position_tercile=1 + 3 * pos // total, popularity_tier=tier,
        ground_truth=gt,
    )


# ---------------------------------------------------------------- Constitutions
WORDNUM = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7}
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI"}


def parse_us():
    txt = (RAW / "us_constitution.txt").read_text(encoding="utf-8-sig").replace("\r", "")
    body = txt.split("*** START OF THE PROJECT GUTENBERG EBOOK")[1].split("*** END OF THE PROJECT GUTENBERG EBOOK")[0]
    m = re.search(r"^Article 1\s*$", body, re.M)
    body = body[m.start():]
    end = re.search(r"^\s*(?:done in Convention|Done in Convention)", body, re.M | re.I)
    if end:
        body = body[:end.start()]
    art_re = re.compile(r"^(?:Article|ARTICLE)\s+(\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN)\s*$", re.M)
    arts = list(art_re.finditer(body))
    units = []
    for ai, am in enumerate(arts):
        a = am.group(1)
        a = int(a) if a.isdigit() else WORDNUM[a]
        chunk = body[am.end(): arts[ai + 1].start() if ai + 1 < len(arts) else len(body)]
        secs = list(re.finditer(r"^Section (\d+)\.\s+", chunk, re.M))
        if not secs:
            units.append((a, None, collapse_ws(clean_quotes(chunk))))
        for si, sm in enumerate(secs):
            t = chunk[sm.end(): secs[si + 1].start() if si + 1 < len(secs) else len(chunk)]
            units.append((a, int(sm.group(1)), collapse_ws(clean_quotes(t))))
    return units


def parse_tn():
    txt = (RAW / "tn_constitution.txt").read_text(encoding="utf-8")
    body = txt[re.search(r"^ARTICLE I\.\s*$", txt, re.M).start():]
    body = body.split("\nSchedule.\n")[0]
    roman_to = {v: k for k, v in ROMAN.items()}
    arts = list(re.finditer(r"^ARTICLE ([IVX]+)\.\s*$", body, re.M))
    units = []
    for ai, am in enumerate(arts):
        a = roman_to[am.group(1)]
        chunk = body[am.end(): arts[ai + 1].start() if ai + 1 < len(arts) else len(body)]
        secs = list(re.finditer(r"^Section (\d+[a-z]?)\.\s+", chunk, re.M))
        for si, sm in enumerate(secs):
            t = chunk[sm.end(): secs[si + 1].start() if si + 1 < len(secs) else len(chunk)]
            # drop trailing sub-heading lines such as "Legislative Department."
            t = re.sub(r"\n[A-Z][A-Za-z ]+\.\s*$", "", t.strip())
            t = collapse_ws(clean_quotes(t))
            if "[Deleted]" in t or "[Repealed" in t:
                continue
            units.append((a, sm.group(1), t))
    return units


def legal_items(n, rng):
    us = [u for u in parse_us() if word_count(u[2]) >= 12]
    tn = [u for u in parse_tn() if word_count(u[2]) >= 12]
    n_us = n // 2
    items = []
    for src, units, k, tier in (("us", us, n_us, "famous"), ("tn", tn, n - n_us, "obscure")):
        total = len(units)
        for terc in range(3):
            pool = list(range(terc * total // 3, (terc + 1) * total // 3))
            take = k // 3 + (1 if terc < k % 3 else 0)
            for i in sorted(rng.sample(pool, take)):
                items.append(make_legal(src, units[i], i, total, tier))
    return items


def make_legal(src, unit, pos, total, tier):
    a, s, text = unit
    loc = f"Article {ROMAN[a]}" + (f", Section {s}" if s is not None else "")
    truncated = word_count(text) > MAX_WORDS
    gt = first_n_words(text, MAX_WORDS) if truncated else text
    if src == "us":
        meta = dict(title="the Constitution of the United States", author="the Philadelphia Convention",
                    edition="text of 1787 as signed, without amendments")
    else:
        meta = dict(title="the Constitution of the State of Tennessee", author="the Tennessee Constitutional Convention",
                    edition="1870 Constitution, as amended (current official text)")
    return dict(
        text_id=f"{src}_constitution", item_id=f"{src}const-{a}-{s if s is not None else 0}",
        item_noun="", locator=loc,
        span=f"first {MAX_WORDS} words" if truncated else "full section",
        span_instruction=(f"Reproduce only the opening of this provision (about the first {MAX_WORDS} words)."
                          if truncated else "Reproduce the full provision."),
        item_type="numbered_prose_unit", position_tercile=1 + 3 * pos // total, popularity_tier=tier,
        ground_truth=gt, **meta,
    )


# ---------------------------------------------------------------- critical elements
NUMWORDS = set("one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty sixty seventy eighty ninety hundred thousand first second third fourth fifth sixth seventh".split())


def critical_elements(gt):
    """Numbers, number words, and proper nouns (capitalised words not at a sentence start).
    Prose-only heuristic for the pilot; math texts will need formula/variable extraction."""
    out = []
    for m in re.finditer(r"[A-Za-z']+|\d[\d,.$%]*", gt):
        t = m.group(0).strip("'")
        if not t:
            continue
        before = re.sub(r"[\s\"'(]+$", "", gt[:m.start()])
        at_start = before == "" or before[-1] in ".?!:" or "\n" in gt[len(before):m.start()]
        if re.match(r"\d", t) or t.lower() in NUMWORDS:
            out.append(t.lower())
        elif t[0].isupper() and not at_start and t not in ("I", "O"):
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-text", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    items = []
    items += kjv_items(args.per_text, rng)
    items += sonnet_items(args.per_text, rng)
    items += tractatus_items(args.per_text, rng)
    items += legal_items(args.per_text, rng)
    ITEMS.mkdir(parents=True, exist_ok=True)
    seen = set()
    for it in items:
        assert it["item_id"] not in seen, it["item_id"]
        seen.add(it["item_id"])
        it["ground_truth_sha256"] = sha256(it["ground_truth"])
        it["length_words"] = word_count(it["ground_truth"])
        ce = critical_elements(it["ground_truth"])
        it["critical_elements"] = ce
        it["n_critical_elements"] = len(ce)
        it["prompt_variant"] = rng.choice(["a", "b"])
    with open(ITEMS / "pilot_ground_truth.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    cols = ["item_id", "text_id", "edition", "item_noun", "locator", "span", "item_type", "position_tercile",
            "popularity_tier", "ground_truth_sha256", "length_words", "n_critical_elements", "prompt_variant"]
    with open(ITEMS / "pilot_items.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for it in items:
            w.writerow(it)
    from collections import Counter
    print(f"{len(items)} items")
    print(Counter((i["text_id"], i["popularity_tier"]) for i in items))


if __name__ == "__main__":
    main()
