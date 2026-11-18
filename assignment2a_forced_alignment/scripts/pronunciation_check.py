"""Use the aligner as a pronunciation checker for the ISLE minimal pairs ("I said BAD not BED").

Forced alignment normally takes the dictionary's word for granted. Here each target word also gets the vowel of
its partner (bad -> B EH1 D, bed -> B AE1 D, white -> W EY1 T, ...), and the acoustic model decides which fits:

  free       one dictionary holding both vowels of every target word with equal prior; the TextGrid shows which
             pronunciation the aligner chose
  canonical  only the intended vowels
  swap1      the first target word of each sentence ("white", "new", "bad") gets its partner's vowel
  swap2      the second target word ("bait", "no", "bed") gets its partner's vowel

For each target word, LLR = log-likelihood(swapped) - log-likelihood(canonical), summed over the utterance's
frames. Positive: the audio fits the partner's vowel better, i.e. the word may have been mispronounced towards its
partner. This is the idea behind goodness-of-pronunciation (GOP) scores, done with MFA itself.

    python scripts/pronunciation_check.py [outputs dir]    (needs `mfa` on PATH; MFA_TEMP as in run_pipeline.py)
"""
import csv
import glob
import os
import re
import shutil
import subprocess
import sys

from praatio import textgrid

OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs"
BASE = os.path.join(OUT, "pron_check")
ACOUSTIC = "english_us_arpa"
# (first, second) target word of each ISLE sentence, with the intended pronunciations
PAIRS = {"01": (("white", "W AY1 T"), ("bait", "B EY1 T")),
         "02": (("new", "N UW1"), ("no", "N OW1")),
         "03": (("bad", "B AE1 D"), ("bed", "B EH1 D"))}
VOWELS = r"(AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)\d"


def swapped(pron, partner):
    """The pronunciation with the partner word's vowel."""
    v = re.search(VOWELS, partner).group(0)
    return re.sub(VOWELS, v, pron, count=1)


def pretrained_entry(word):
    path = os.path.join(os.environ.get("MFA_ROOT_DIR", os.path.expanduser("~/Documents/MFA")),
                        "pretrained_models", "dictionary", "english_us_arpa.dict")
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if parts[0] == word:
            return parts[-1]  # the pronunciation is the last column (probabilities come before it)
    raise KeyError(word)


def write_dict(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for w, p in entries:
            f.write(f"{w}\t{p}\n")


def mfa_align(corpus, dictionary, out):
    shutil.rmtree(out, ignore_errors=True)
    extra = ["--temporary_directory", os.environ["MFA_TEMP"]] if os.environ.get("MFA_TEMP") else []
    cmd = ["mfa", "align", corpus, dictionary, ACOUSTIC, out, "--clean", "--output_format", "long_textgrid"] + extra
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def loglik(out):
    """{utterance: total log-likelihood} from MFA's alignment_analysis.csv (per-frame average x frames)."""
    res = {}
    for r in csv.DictReader(open(os.path.join(out, "alignment_analysis.csv"), encoding="utf-8")):
        frames = round((float(r["end"]) - float(r["begin"])) / 0.01)
        res[r["file"]] = float(r["overall_log_likelihood"]) * frames
    return res


def main():
    corpus = os.path.join(BASE, "corpus")
    shutil.rmtree(corpus, ignore_errors=True)
    shutil.copytree(os.path.join(OUT, "corpus", "isle_sess0131"), os.path.join(corpus, "isle_sess0131"),
                    ignore=shutil.ignore_patterns("*.tsv"))
    common = [(w, pretrained_entry(w)) for w in ("i", "said", "not")]
    first = [p[0] for p in PAIRS.values()]
    second = [p[1] for p in PAIRS.values()]
    partner = {a[0]: b[1] for a, b in PAIRS.values()} | {b[0]: a[1] for a, b in PAIRS.values()}
    dicts = {
        "free": common + [(w, p) for w, p in first + second] + [(w, swapped(p, partner[w])) for w, p in first + second],
        "canonical": common + first + second,
        "swap1": common + [(w, swapped(p, partner[w])) for w, p in first] + second,
        "swap2": common + first + [(w, swapped(p, partner[w])) for w, p in second],
    }
    ll = {}
    for name, entries in dicts.items():
        d = os.path.join(BASE, f"{name}.dict")
        write_dict(d, entries)
        mfa_align(corpus, d, os.path.join(BASE, name))
        ll[name] = loglik(os.path.join(BASE, name))

    rows = [("utterance", "word", "intended", "partner_vowel_version", "free_choice", "llr_partner_minus_intended")]
    for key, ((w1, p1), (w2, p2)) in PAIRS.items():
        path = glob.glob(os.path.join(BASE, "free", "*", f"*BLOCKD02_{key}_*.TextGrid"))[0]
        utt = os.path.basename(path)[:-len(".TextGrid")]
        tg = textgrid.openTextgrid(path, includeEmptyIntervals=False)
        phones = tg.getTier("phones").entries
        for (w, p), swap_run in (((w1, p1), "swap1"), ((w2, p2), "swap2")):
            span = [e for e in tg.getTier("words").entries if e[2] == w][0]
            chosen = " ".join(x[2] for x in phones if x[0] >= span[0] - 1e-6 and x[1] <= span[1] + 1e-6)
            llr = ll[swap_run][utt] - ll["canonical"][utt]
            rows.append((utt, w, p, swapped(p, partner[w]), chosen, round(llr, 1)))
    os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
    with open(os.path.join(OUT, "analysis", "pronunciation_check.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    for r in rows:
        print(" | ".join(map(str, r)))


if __name__ == "__main__":
    main()
