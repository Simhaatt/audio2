"""Numbers behind the report: what each setup produced, where the setups disagree, and what the aligner did with
the non-native minimal pairs.

Writes to outputs/analysis/:
  summary.csv            per setup: words, phones, words aligned as unknown (spn), pauses, phone durations
  word_agreement.csv     per pair of setups: how far apart the same word's start/end times are
  word_differences.csv   every word token with its start/end in each setup (to find the largest disagreements)
  pauses.csv             silences of at least 100 ms inside an utterance, with the punctuation next to them
  short_phones.csv       phones of 30 ms or less (three frames or fewer), with their word
  minimal_pairs.csv      ISLE: vowel label, duration and formants (F1/F2, Praat Burg) of each target word

    python scripts/analyze.py [outputs dir]
"""
import csv
import difflib
import glob
import itertools
import json
import os
import re
import sys

import numpy as np
import parselmouth
from praatio import textgrid

OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs"
A = os.path.join(OUT, "analysis")
SIL = {"", "sil", "sp", "spn", "<eps>"}
MIN_PHONE = 0.030 + 1e-6  # 30 ms or less = three 10 ms frames or fewer: phones squeezed to (near) nothing
PAIRS = {"01": ("white", "bait"), "02": ("new", "no"), "03": ("bad", "bed")}
CEILING = {"bu_f2b": 5500, "isle_sess0131": 5000}  # Praat's advice: 5500 Hz for a female voice, 5000 for male


def load(setup):
    res = {}
    for p in sorted(glob.glob(os.path.join(OUT, "textgrids", setup, "*", "*.TextGrid"))):
        tg = textgrid.openTextgrid(p, includeEmptyIntervals=True)
        spk = os.path.basename(os.path.dirname(p))
        utt = os.path.basename(p)[:-len(".TextGrid")]
        res[(spk, utt)] = {n: [(s, e, t.strip()) for s, e, t in tg.getTier(n).entries] for n in tg.tierNames}
    return res


def transcript_tokens(text):
    """(word, punctuation after it) in transcript order, split at hyphens the way MFA splits words."""
    out = []
    for raw in text.split():
        punct = raw[-1] if raw[-1] in ",.;:?!" else ""
        parts = [p for p in re.sub(r"[^\w'-]", "", raw).lower().split("-") if p]
        out += [(p, "") for p in parts[:-1]] + ([(parts[-1], punct)] if parts else [])
    return out


def words(tiers):
    return [(s, e, t) for s, e, t in tiers["words"] if t.lower() not in SIL]


def phones_in(tiers, s, e):
    return [(ps, pe, p) for ps, pe, p in tiers["phones"] if ps >= s - 1e-6 and pe <= e + 1e-6]


def write(name, rows):
    os.makedirs(A, exist_ok=True)
    with open(os.path.join(A, name), "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def main():
    setups = sorted(os.listdir(os.path.join(OUT, "textgrids")))
    data = {s: load(s) for s in setups}
    report = {}

    # 1. what each setup produced
    rows = [("setup", "utterances", "words", "words_as_spn", "phones", "phone_ms_median", "phones_30ms_or_less",
             "pauses_100ms_plus", "pause_s_total")]
    for s in setups:
        nw = nspn = nph = nmin = npause = 0
        durs, pause_total = [], 0.0
        for tiers in data[s].values():
            w = words(tiers)
            nw += len(w)
            ph = [(a, b, p) for a, b, p in tiers["phones"] if p.lower() not in SIL]
            nspn += sum(1 for a, b, p in tiers["phones"] if p == "spn")
            nph += len(ph)
            durs += [b - a for a, b, p in ph]
            nmin += sum(1 for a, b, p in ph if b - a <= MIN_PHONE)
            inner = [(a, b) for a, b, t in tiers["words"] if t.lower() in SIL and a > w[0][0] and b < w[-1][1]]
            long = [(a, b) for a, b in inner if b - a >= 0.1]
            npause += len(long)
            pause_total += sum(b - a for a, b in long)
        rows.append((s, len(data[s]), nw, nspn, nph, round(1000 * float(np.median(durs)), 1), nmin, npause,
                     round(pause_total, 2)))
    write("summary.csv", rows)
    report["summary"] = rows

    # 2. agreement of word boundaries between setups (same transcript -> same word sequence)
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    tokens = []
    for k in keys:
        seqs = {s: words(data[s][k]) for s in setups}
        n = min(len(v) for v in seqs.values())
        for i in range(n):
            tokens.append((k, i, seqs[setups[0]][i][2], {s: seqs[s][i][:2] for s in setups}))
    write("word_differences.csv", [("speaker", "utterance", "index", "word")
                                   + tuple(f"{s}_{x}" for s in setups for x in ("start", "end"))]
          + [(k[0], k[1], i, w) + tuple(round(v[s][j], 3) for s in setups for j in (0, 1))
             for k, i, w, v in tokens])
    agree = [("setup_a", "setup_b", "words", "median_abs_diff_ms", "mean_abs_diff_ms", "within_20ms_pct",
              "within_50ms_pct", "largest_ms", "largest_word")]
    for a, b in itertools.combinations(setups, 2):
        d, worst = [], (0, "")
        for k, i, w, v in tokens:
            for j in (0, 1):
                x = abs(v[a][j] - v[b][j])
                d.append(x)
                if x > worst[0]:
                    worst = (x, f"{w} ({k[1]}, {'start' if j == 0 else 'end'})")
        d = np.array(d)
        agree.append((a, b, len(tokens), round(1000 * float(np.median(d)), 1), round(1000 * float(d.mean()), 1),
                      round(100 * float((d <= 0.02).mean()), 1), round(100 * float((d <= 0.05).mean()), 1),
                      round(1000 * worst[0]), worst[1]))
    write("word_agreement.csv", agree)
    report["agreement"] = agree

    # 3. pauses and the punctuation around them (news clips)
    pause_rows = [("setup", "utterance", "start", "end", "ms", "word_before", "punctuation_after_word_before")]
    for s in setups:
        for (spk, utt), tiers in data[s].items():
            toks = transcript_tokens(open(os.path.join(OUT, "corpus", spk, utt + ".lab"), encoding="utf-8").read())
            w = words(tiers)
            # map each aligned word to its transcript token (MFA splits hyphenated words, so match sequences)
            sm = difflib.SequenceMatcher(None, [x[2].lower() for x in w], [t[0] for t in toks], autojunk=False)
            punct_of = {}
            for blk in sm.get_matching_blocks():
                for j in range(blk.size):
                    punct_of[blk.a + j] = toks[blk.b + j][1]
            for a, b, t in tiers["words"]:
                if t.lower() in SIL and b - a >= 0.1 and a > w[0][0] and b < w[-1][1]:
                    idx = sum(1 for x in w if x[1] <= a + 1e-6) - 1
                    pause_rows.append((s, utt, round(a, 3), round(b, 3), round(1000 * (b - a)),
                                       w[idx][2] if idx >= 0 else "", punct_of.get(idx, "")))
    write("pauses.csv", pause_rows)

    # 4. phones at the minimum length
    short = [("setup", "utterance", "word", "phone", "start", "ms")]
    for s in setups:
        for (spk, utt), tiers in data[s].items():
            for ws, we, w in words(tiers):
                for ps, pe, p in phones_in(tiers, ws, we):
                    if p.lower() not in SIL and pe - ps <= MIN_PHONE:
                        short.append((s, utt, w, p, round(ps, 3), round(1000 * (pe - ps))))
    write("short_phones.csv", short)

    # 5. ISLE minimal pairs: which vowel the aligner labelled, and what the speaker actually produced
    mp = [("setup", "utterance", "word", "vowel_label", "start", "ms", "F1_Hz", "F2_Hz")]
    for s in setups:
        for (spk, utt), tiers in data[s].items():
            m = re.search(r"BLOCKD02_(\d+)_", utt)
            if not m or m.group(1) not in PAIRS:
                continue
            snd = parselmouth.Sound(os.path.join(OUT, "corpus", spk, utt + ".wav"))
            fm = snd.to_formant_burg(maximum_formant=CEILING.get(spk, 5500))
            for ws, we, w in words(tiers):
                if w.lower() not in PAIRS[m.group(1)]:
                    continue
                for ps, pe, p in phones_in(tiers, ws, we):
                    if re.match(r"^(AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)\d?$", p) or \
                            re.search(r"[aeiouæɑɐɛɪʊɔəɜʉɒ]", p):
                        ts = np.linspace(ps + 0.25 * (pe - ps), pe - 0.25 * (pe - ps), 7)
                        f1 = np.nanmedian([fm.get_value_at_time(1, t) for t in ts])
                        f2 = np.nanmedian([fm.get_value_at_time(2, t) for t in ts])
                        mp.append((s, utt, w, p, round(ps, 3), round(1000 * (pe - ps)), round(f1), round(f2)))
    write("minimal_pairs.csv", mp)
    report["minimal_pairs"] = mp

    json.dump(report, open(os.path.join(A, "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for name in ("summary", "agreement", "minimal_pairs"):
        print(f"== {name}")
        for r in report[name]:
            print("  " + " | ".join(map(str, r)))
    print(f"pauses >= 100 ms: {len(pause_rows) - 1} rows, short phones: {len(short) - 1} rows -> {A}")


if __name__ == "__main__":
    main()
