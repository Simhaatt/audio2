"""TESS: find the files, read the labels from the file names, build the transcripts and the splits.

TESS (Toronto emotional speech set): two actresses, OAF (aged 64) and YAF (aged 26), each say "Say the word ___"
with 200 target words in 7 emotions -> 2,800 clips. File names are <speaker>_<word>_<emotion>.wav, with the
emotion codes angry, disgust, fear, happy, neutral, ps (pleasant surprise) and sad. Some copies of the dataset
contain the same files twice in nested folders; files are de-duplicated by name.

Splits (all keep the 7 emotions balanced):
  word       the 200 words are split 140/30/30 into train/val/test, both speakers in every part. Test sentences
             are never seen in training, so a text model cannot memorise them.
  oaf2yaf    train (and validate, on held-out words) on OAF, test on all of YAF: a voice never heard in training.
  yaf2oaf    the reverse.
"""
import json
import os
import random
import re
from collections import Counter

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]
EMOTION_NAMES = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "neutral": "neutral",
                 "ps": "pleasant surprise", "sad": "sad"}
SPLITS = ["word", "oaf2yaf", "yaf2oaf"]
_ALIASES = {"pleasant_surprise": "ps", "pleasant_surprised": "ps", "surprise": "ps", "surprised": "ps",
            "fearful": "fear", "disgusted": "disgust", "anger": "angry"}


def transcript(word):
    """The spoken sentence. TESS gives the target word; the carrier phrase is the same in every clip."""
    return f"Say the word {word}."


def find_tess(root):
    """All TESS clips under root as dicts: id, path, speaker, word, emotion, label, text."""
    items, seen, skipped = [], set(), []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if not f.lower().endswith(".wav"):
                continue
            stem = os.path.splitext(f)[0]
            parts = stem.split("_")
            if len(parts) < 3:
                skipped.append(f)
                continue
            spk, emo = parts[0].upper(), parts[-1].lower()
            emo = _ALIASES.get(emo, emo)
            word = "_".join(parts[1:-1]).lower()
            uid = f"{spk}_{word}_{emo}"
            if emo not in EMOTIONS or spk not in ("OAF", "YAF"):
                skipped.append(f)
                continue
            if uid in seen:
                continue
            seen.add(uid)
            items.append({"id": uid, "path": os.path.join(dirpath, f), "speaker": spk, "word": word,
                          "emotion": emo, "label": EMOTIONS.index(emo), "text": transcript(word)})
    items.sort(key=lambda x: x["id"])
    if skipped:
        print(f"skipped {len(skipped)} files with unexpected names, e.g. {skipped[:3]}")
    return items


def describe(items):
    c = Counter((x["speaker"], x["emotion"]) for x in items)
    words = Counter(x["word"] for x in items)
    print(f"{len(items)} clips, {len(words)} words, speakers {sorted({x['speaker'] for x in items})}")
    print("clips per speaker x emotion:", dict(sorted(c.items())))
    odd = {w: n for w, n in words.items() if n != 14}
    if odd:
        print(f"{len(odd)} words do not have exactly 14 clips (2 speakers x 7 emotions), e.g.",
              dict(list(odd.items())[:5]))


def _split_words(words, seed, fracs=(0.7, 0.15, 0.15)):
    words = sorted(words)
    random.Random(seed).shuffle(words)
    n_tr = round(fracs[0] * len(words))
    n_va = round(fracs[1] * len(words))
    return set(words[:n_tr]), set(words[n_tr:n_tr + n_va]), set(words[n_tr + n_va:])


def make_splits(items, seed=0):
    """{split: {"train": [ids], "val": [ids], "test": [ids]}}. The word partition is fixed by `seed`."""
    out = {}
    tr, va, te = _split_words({x["word"] for x in items}, seed)
    out["word"] = {"train": [x["id"] for x in items if x["word"] in tr],
                   "val": [x["id"] for x in items if x["word"] in va],
                   "test": [x["id"] for x in items if x["word"] in te]}
    for name, src, dst in (("oaf2yaf", "OAF", "YAF"), ("yaf2oaf", "YAF", "OAF")):
        src_items = [x for x in items if x["speaker"] == src]
        tr, va, _ = _split_words({x["word"] for x in src_items}, seed, (0.85, 0.15, 0.0))
        out[name] = {"train": [x["id"] for x in src_items if x["word"] in tr],
                     "val": [x["id"] for x in src_items if x["word"] in va],
                     "test": [x["id"] for x in items if x["speaker"] == dst]}
    return out


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean_text(text):
    """Text preprocessing: lower case, punctuation removed, single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()
