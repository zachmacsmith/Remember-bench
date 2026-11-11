"""Shared paths and helpers."""
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ITEMS = ROOT / "data" / "items"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def clean_quotes(s: str) -> str:
    """Map typographic quotes/dashes to ASCII so ground truth and outputs compare fairly."""
    table = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "—": "--", "–": "-", " ": " ",
    }
    for k, v in table.items():
        s = s.replace(k, v)
    return s


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def first_n_words(s: str, n: int) -> str:
    w = s.split()
    return " ".join(w[:n])


def word_count(s: str) -> int:
    return len(s.split())


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(name="pilot"):
    with open(CONFIGS / f"{name}.json") as f:
        return json.load(f)


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)
