"""Features for the speaker-augmentation experiment (cross-speaker splits).

    python scripts/extract_augmented.py --tess <folder with the TESS wavs> --out features_aug [--smoke]

Writes metadata.json and splits.json (the same as scripts/extract_features.py: same files, same seed), and
  speech_wavlmmean.pt      every clip: mean of WavLM Base+'s 13 hidden layers (T x 768)
  speech_logmel.pt         every clip: 64-band log-mel (T x 64)
  speech_wavlmmean_aug.pt  the training clips of both cross-speaker splits in 4 augmented versions
  speech_logmel_aug.pt     the same for log-mel
Augmented keys are '<clip id>|<augmentation>' with augmentations pitch-3, pitch+3, speed0.9 and speed1.1
(common/features.py). Validation and test clips are never augmented. The layer mean keeps the files 13 times
smaller than all layers; the speech model's learned layer weights were close to uniform anyway.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.data import describe, find_tess, make_splits, save_json  # noqa: E402
from common.features import AUGMENTATIONS, load_audio, load_ssl, logmel, ssl_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tess", required=True)
    ap.add_argument("--out", default="features_aug")
    ap.add_argument("--ssl", default="microsoft/wavlm-base-plus")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(a.out, exist_ok=True)
    items = find_tess(a.tess)
    describe(items)
    splits = make_splits(items, a.split_seed)
    save_json(items, os.path.join(a.out, "metadata.json"))
    save_json(splits, os.path.join(a.out, "splits.json"))
    fe, ssl = load_ssl(a.ssl, dev, a.smoke)

    def feats(wav):
        return ssl_features(wav, fe, ssl, dev).float().mean(0).half(), torch.from_numpy(logmel(wav))

    t0 = time.time()
    wm, lm = {}, {}
    for x in items:
        wm[x["id"]], lm[x["id"]] = feats(load_audio(x["path"]))
    torch.save(wm, os.path.join(a.out, "speech_wavlmmean.pt"))
    torch.save(lm, os.path.join(a.out, "speech_logmel.pt"))
    print(f"original clips: {len(wm)} ({time.time() - t0:.0f} s)", flush=True)

    train_ids = set(splits["oaf2yaf"]["train"]) | set(splits["yaf2oaf"]["train"])
    wm_aug, lm_aug = {}, {}
    for n, x in enumerate([x for x in items if x["id"] in train_ids], 1):
        for kind in AUGMENTATIONS:
            k = f"{x['id']}|{kind}"
            wm_aug[k], lm_aug[k] = feats(load_audio(x["path"], augmentation=kind))
        if n % 500 == 0:
            print(f"augmented {n}/{len(train_ids)} clips ({time.time() - t0:.0f} s)", flush=True)
    torch.save(wm_aug, os.path.join(a.out, "speech_wavlmmean_aug.pt"))
    torch.save(lm_aug, os.path.join(a.out, "speech_logmel_aug.pt"))
    print(f"augmented copies: {len(wm_aug)} of {len(train_ids)} training clips, done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
