"""Praat-style pictures of the alignments: waveform, spectrogram, word tier and phone tier on one time axis.

The spectrogram comes from Praat itself (through parselmouth, Praat 6.1), with Praat's default settings:
5 ms Gaussian window, 0-5 kHz. For each setup and utterance it draws
  figures/<setup>/<utt>.png            the whole utterance (long news clips: a 3.5 s stretch, plus a full overview)
  figures/compare/<utt>.png            the same stretch with the phone tiers of every setup stacked, to see
                                       where the acoustic models put boundaries differently

    python scripts/visualize.py [outputs dir]
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import parselmouth  # noqa: E402
from praatio import textgrid  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs"
ZOOM = 3.5  # seconds shown for clips longer than LONG (the news clips); shorter clips are drawn whole
LONG = 8.0
# a different stretch where it shows something worth seeing (F2BJRLP2: the OOV words "Dukakis", "politicize")
WINDOWS = {"F2BJRLP2": (2.4, 5.9)}
SETUP_LABELS = {"english_us_arpa_pretrained_dict": "english_us_arpa, pretrained dictionary",
                "english_us_arpa": "english_us_arpa + G2P dictionary",
                "english_mfa": "english_mfa + G2P dictionary"}
SILENCE = {"", "sil", "sp", "spn"}


def tiers(path):
    tg = textgrid.openTextgrid(path, includeEmptyIntervals=False)
    return {name: tg.getTier(name).entries for name in tg.tierNames}


def draw_tier(ax, entries, t0, t1, label, fontsize=8):
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    for start, end, text in entries:
        if end < t0 or start > t1:
            continue
        colour = "#999999" if text in SILENCE else "black"
        ax.axvline(start, color="#1f5fa8", lw=0.8)
        ax.axvline(end, color="#1f5fa8", lw=0.8)
        mid = (max(start, t0) + min(end, t1)) / 2
        ax.text(mid, 0.5, text, ha="center", va="center", fontsize=fontsize, color=colour, clip_on=True)


def figure(wav, tier_sets, t0, t1, title, path):
    snd = parselmouth.Sound(wav)
    part = snd.extract_part(t0, t1, preserve_times=True)
    spec = part.to_spectrogram(window_length=0.005, maximum_frequency=5000)
    db = 10 * np.log10(np.maximum(spec.values, 1e-12))
    rows = 2 + len(tier_sets)
    fig, axes = plt.subplots(rows, 1, figsize=(min(16, max(8, 3.0 * (t1 - t0))), 1.2 + 1.0 * rows),
                             gridspec_kw={"height_ratios": [1, 2.2] + [0.55] * (rows - 2)}, sharex=True)
    axes[0].plot(part.xs(), part.values[0], lw=0.4, color="black")
    axes[0].set_yticks([])
    axes[0].set_ylabel("wave", rotation=0, ha="right", va="center", fontsize=8)
    vmax = db.max()
    axes[1].pcolormesh(spec.xs(), spec.ys(), db, vmin=vmax - 70, vmax=vmax, cmap="Greys", shading="auto")
    axes[1].set_ylabel("Hz", fontsize=8)
    for ax, (label, entries) in zip(axes[2:], tier_sets):
        draw_tier(ax, entries, t0, t1, label, fontsize=8 if t1 - t0 < 8 else 6)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    setups = sorted(d for d in os.listdir(os.path.join(OUT, "textgrids")))
    utts = sorted({os.path.relpath(p, os.path.join(OUT, "textgrids", s))
                   for s in setups for p in glob.glob(os.path.join(OUT, "textgrids", s, "*", "*.TextGrid"))})
    for rel in utts:
        spk, name = os.path.split(rel)
        utt = name[:-len(".TextGrid")]
        wav = os.path.join(OUT, "corpus", spk, utt + ".wav")
        dur = parselmouth.Sound(wav).duration
        t0, t1 = WINDOWS.get(utt, (0.0, ZOOM if dur > LONG else dur))
        loaded = {s: tiers(os.path.join(OUT, "textgrids", s, rel)) for s in setups
                  if os.path.exists(os.path.join(OUT, "textgrids", s, rel))}
        for s, t in loaded.items():
            figure(wav, [("words", t["words"]), ("phones", t["phones"])], t0, t1,
                   f"{utt}  |  {SETUP_LABELS.get(s, s)}" + (f"  ({t0:.1f}-{t1:.1f} s)" if dur > LONG else ""),
                   os.path.join(OUT, "figures", s, utt + ".png"))
            if dur > LONG:
                figure(wav, [("words", t["words"])], 0, dur, f"{utt}  |  {SETUP_LABELS.get(s, s)}  (whole clip)",
                       os.path.join(OUT, "figures", s, utt + "_full.png"))
        first = next(iter(loaded))
        stack = [(f"words\n({first})", loaded[first]["words"])]
        stack += [(SETUP_LABELS.get(s, s).split(" ")[0] + ("\n(pretrained dict)" if "pretrained" in s else ""),
                   t["phones"]) for s, t in loaded.items()]
        figure(wav, stack, t0, t1, f"{utt}  |  phone boundaries of each setup",
               os.path.join(OUT, "figures", "compare", utt + ".png"))
        print("drew", utt)


if __name__ == "__main__":
    main()
