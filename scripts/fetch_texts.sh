#!/usr/bin/env bash
# Re-download the corpus into data/raw (see data/SOURCES.md). The Tennessee Constitution is not fetched:
# the official PDF URL returned a 404 page, and the committed text was supplied by hand.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
get() { curl -fsSL --retry 4 --retry-delay 2 -o "$1" "$2"; echo "fetched $1"; }
get kjv.txt                  https://www.gutenberg.org/cache/epub/10/pg10.txt
get shakespeare_sonnets.txt  https://www.gutenberg.org/cache/epub/1041/pg1041.txt
get us_constitution.txt      https://www.gutenberg.org/cache/epub/5/pg5.txt
get spinoza_ethics_elwes.txt https://www.gutenberg.org/cache/epub/3800/pg3800.txt
get tractatus_ogden.tex      https://www.gutenberg.org/files/5740/5740-t/5740-t.tex
get tractatus_ogden.pdf      https://www.gutenberg.org/files/5740/5740-pdf.pdf
get axler_ladr4e.pdf         https://linear.axler.net/LADR4e.pdf
get rfc791.txt               https://www.rfc-editor.org/rfc/rfc791.txt
get rfc9110.txt              https://www.rfc-editor.org/rfc/rfc9110.txt
get rfc7323.txt              https://www.rfc-editor.org/rfc/rfc7323.txt
get euclid_heath1908.xml     https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg1799/tlg001/tlg1799.tlg001.perseus-eng2.xml
get openrouter_models.json   https://openrouter.ai/api/v1/models
tmp=$(mktemp -d)
git clone -q https://github.com/rbeezer/fcla "$tmp/fcla" && git -C "$tmp/fcla" checkout -q 347f27fe909b54b26970bcc0fcbf69c7e853c766
git clone -q https://github.com/sarabander/sicp "$tmp/sicp" && git -C "$tmp/sicp" checkout -q bda03f79d6e2e8899ac2b5ca6a3732210e290a79
mkdir -p beezer_fcla sicp
cp "$tmp"/fcla/src/*.xml "$tmp"/fcla/COPYING.txt beezer_fcla/
cp "$tmp"/sicp/sicp-pocket.texi "$tmp"/sicp/LICENSE "$tmp"/sicp/LICENSE.src sicp/
rm -rf "$tmp"
sha256sum -c SHA256SUMS || echo "WARNING: some files changed upstream since the pinned snapshot"
