"""Put the provided data into the layout MFA expects: <corpus>/<speaker>/<utterance>.wav + .lab

- Audio: the three BU Radio News files (F2BJRLP*.wav) are NIST SPHERE files with a .wav name (header "NIST_1A",
  16-bit little-endian PCM, 16 kHz, mono). MFA's audio reader expects RIFF/WAV, so they are rewritten as RIFF
  with the same samples. The ISLE files are already RIFF and are copied unchanged.
- Text: transcripts are joined onto one line and lightly normalised so every token is a word the speaker says.
  Every change is written to normalisation_log.tsv. Case and ordinary punctuation are left to MFA.
- Speakers: one folder per speaker (BU speaker f2b, ISLE session 0131), so MFA's speaker adaptation sees each
  speaker's files together.

- Optional trailing-silence trim (--trim_db): cut the recording 150 ms after the last 10 ms frame whose energy is
  within trim_db of the loudest frame. The start is kept, so times stay comparable with the untrimmed corpus.
  Used for the english_mfa_trimmed setup: without it english_mfa stretches word-final stops over the silence.

    python scripts/prepare_corpus.py [data dir] [output corpus dir] [--trim_db 25]
"""
import argparse
import csv
import os
import re
import shutil
import sys
import wave

_ap = argparse.ArgumentParser()
_ap.add_argument("data", nargs="?", default="data")
_ap.add_argument("out", nargs="?", default=os.path.join("outputs", "corpus"))
_ap.add_argument("--trim_db", type=float, default=None)
_args = _ap.parse_args()
DATA, OUT, TRIM_DB = _args.data, _args.out, _args.trim_db

# (pattern, replacement, reason), applied in order
RULES = [
    (r"(\w+)- (\w+)", r"\1 \2", "number split across a hyphen and a space ('seventy- six'): two spoken words"),
    (r"\bS\.J\.C\.'s\b", "S J C's", "acronym read letter by letter (Supreme Judicial Court)"),
    (r"\bWBUR's\b", "W B U R's", "radio station call letters read letter by letter"),
]


def speaker_of(utt):
    if utt.startswith("F2B"):
        return "bu_f2b"
    m = re.match(r"ISLE_SESS(\d+)_", utt)
    return f"isle_sess{m.group(1)}" if m else "unknown"


def read_sphere(path):
    """Header fields and raw sample bytes of a NIST SPHERE file (uncompressed PCM only)."""
    raw = open(path, "rb").read()
    head_len = int(raw[8:16].decode("ascii").strip())
    fields = {}
    for line in raw[16:head_len].decode("ascii", "replace").splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3:
            fields[parts[0]] = parts[2].strip()
        if line.strip() == "end_head":
            break
    assert fields.get("sample_coding", "pcm") == "pcm", f"{path}: compressed SPHERE is not supported"
    assert fields.get("sample_byte_format") == "01", f"{path}: expected little-endian samples"
    return fields, raw[head_len:]


def write_wav(path, rate, width, channels, frames):
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(frames)


def trim_trailing(path, trim_db, margin=0.15):
    """Cut the file `margin` s after the last 10 ms frame within trim_db of the loudest frame."""
    import numpy as np
    with wave.open(path) as w:
        params, x = w.getparams(), np.frombuffer(w.readframes(w.getnframes()), np.int16)
    hop = params.framerate // 100
    db = 10 * np.log10(np.array([np.mean(x[i:i + hop].astype(float) ** 2) + 1e-9
                                 for i in range(0, len(x) - hop + 1, hop)]))
    end = min(len(x), int((np.where(db > db.max() - trim_db)[0][-1] + 1) * hop + margin * params.framerate))
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(x[:end].tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    log = []
    summary = []
    for name in sorted(os.listdir(os.path.join(DATA, "wav"))):
        utt, _ = os.path.splitext(name)
        src = os.path.join(DATA, "wav", name)
        spk_dir = os.path.join(OUT, speaker_of(utt))
        os.makedirs(spk_dir, exist_ok=True)
        dst = os.path.join(spk_dir, utt + ".wav")
        if open(src, "rb").read(7) == b"NIST_1A":
            f, pcm = read_sphere(src)
            n = int(f["sample_count"])
            write_wav(dst, int(f["sample_rate"]), int(f["sample_n_bytes"]), int(f["channel_count"]),
                      pcm[: n * int(f["sample_n_bytes"])])
            fmt = "NIST SPHERE -> RIFF"
        else:
            shutil.copyfile(src, dst)
            fmt = "RIFF (copied)"
        if TRIM_DB is not None:
            trim_trailing(dst, TRIM_DB)
            fmt += f", trailing silence trimmed ({TRIM_DB:g} dB)"
        with wave.open(dst) as w:
            dur = w.getnframes() / w.getframerate()
            rate = w.getframerate()

        tname = [t for t in os.listdir(os.path.join(DATA, "transcripts")) if os.path.splitext(t)[0] == utt][0]
        text = open(os.path.join(DATA, "transcripts", tname), encoding="utf-8").read()
        text = re.sub(r"\s+", " ", text).strip()
        for pat, rep, why in RULES:
            for m in re.finditer(pat, text):
                log.append((utt, m.group(0), re.sub(pat, rep, m.group(0)), why))
            text = re.sub(pat, rep, text)
        open(os.path.join(spk_dir, utt + ".lab"), "w", encoding="utf-8").write(text + "\n")
        summary.append((utt, speaker_of(utt), fmt, rate, round(dur, 2), len(text.split())))

    with open(os.path.join(OUT, "normalisation_log.tsv"), "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerows([("utterance", "original", "normalised", "reason")] + log)
    with open(os.path.join(OUT, "corpus_summary.tsv"), "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerows([("utterance", "speaker", "audio", "rate", "seconds", "words")]
                                                + summary)
    for row in summary:
        print("\t".join(map(str, row)))
    print(f"{len(log)} normalisations -> {os.path.join(OUT, 'normalisation_log.tsv')}")


if __name__ == "__main__":
    main()
