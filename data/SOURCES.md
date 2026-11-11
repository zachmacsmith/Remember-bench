# Corpus sources (downloaded 2026-11-11)

All files in `data/raw/` are public domain or openly licensed. Checksums: `data/raw/SHA256SUMS`.
Copyrighted texts in the spec (Friedberg–Insel–Spence 5e, Griffiths 4e) are **not** here: ground truth
must come from legally owned copies and stay in a private store (spec, Legal section).

| File | Text | Pinned edition | Source | License / status | In pilot |
|---|---|---|---|---|---|
| `kjv.txt` | King James Bible | 1769 Oxford standard text, Project Gutenberg #10 (updated 2024-10-29) | gutenberg.org/cache/epub/10 | Public domain (US) | yes |
| `shakespeare_sonnets.txt` | Shakespeare, Sonnets | Project Gutenberg #1041 (modern spelling) | gutenberg.org/cache/epub/1041 | Public domain | yes |
| `tractatus_ogden.tex`, `tractatus_ogden.pdf` | Wittgenstein, Tractatus | C. K. Ogden translation, 1922; Project Gutenberg #5740 (TeX source + PDF) | gutenberg.org/files/5740 | Public domain in the US (1922 publication) | yes |
| `us_constitution.txt` | US Constitution | Project Gutenberg #5 (1787 text, no amendments) | gutenberg.org/cache/epub/5 | Public domain | yes |
| `tn_constitution.txt` | Tennessee Constitution | 1870 Constitution as amended (includes Art. I §36, 2014) | Pasted by the user; the official capitol.tn.gov PDF URL returned a 404 page. Verify against an official copy before the full run. | Public domain (government work) | yes |
| `axler_ladr4e.pdf` | Axler, *Linear Algebra Done Right* 4e | Official PDF | linear.axler.net/LADR4e.pdf | Open access, CC BY-NC 4.0 | no (math needs manual LaTeX transcription) |
| `beezer_fcla/*.xml` | Beezer, *A First Course in Linear Algebra* | PreTeXt source, github.com/rbeezer/fcla @ 347f27fe | GitHub | GFDL 1.2+ (`COPYING.txt`) | no |
| `sicp/sicp-pocket.texi` | Abelson & Sussman, *SICP* 2e | Texinfo of the MIT 2e text, github.com/sarabander/sicp @ bda03f79 | GitHub | Text CC BY-SA 4.0 (see `LICENSE`, `LICENSE.src`) | no |
| `euclid_heath1908.xml` | Euclid, *Elements* | Heath 1908, *The Thirteen Books of Euclid's Elements* (Perseus TEI, `perseus-eng2`) | github.com/PerseusDL/canonical-greekLit | Public domain text; Perseus markup CC BY-SA | no |
| `spinoza_ethics_elwes.txt` | Spinoza, *Ethics* | R. H. M. Elwes translation, Project Gutenberg #3800 | gutenberg.org | Public domain | no |
| `rfc791.txt`, `rfc9110.txt`, `rfc7323.txt` | IETF RFCs (IP; HTTP Semantics; obscure: TCP extensions for high performance) | RFC Editor canonical text | rfc-editor.org | IETF Trust; reproduction permitted | no |
| `openrouter_models.json` | Model list and prices (cost estimates) | snapshot 2026-11-11 | openrouter.ai/api/v1/models | n/a | pricing |

## Known edition caveats

- **US Constitution:** Gutenberg #5's spelling and capitalization are not guaranteed to match the National Archives transcription, which is what models most likely memorized. Scoring normalization (lowercase, no punctuation) absorbs most of this. Swap in the archives.gov transcription before the full run.
- **Sonnets:** Gutenberg's modern-spelling text. Elisions ("deserv'd") are close to standard editions, not to the 1609 quarto orthography. The C2 prompt names this edition.
- **Tractatus:** Ogden only. The Pears–McGuinness translation (1961) is copyrighted and common online, so some "wrong" outputs may be correct Pears–McGuinness recall. That is the Tractatus form of edition confusion. Log it as a known risk; the pilot does not tag it.
