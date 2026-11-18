"""Find TESS, write metadata and splits, and cache the features of both modalities.

    python scripts/extract_features.py --tess <folder with the TESS wavs> --out features
    python scripts/extract_features.py --tess <fake data> --out features --smoke   # tiny random encoders

Writes to --out: metadata.json, splits.json, text_tokens.json, speech_wavlm.pt, speech_logmel.pt, text_bert.pt and
speech_wavlm_snr<k>.pt (word-split test clips with white noise at k dB SNR).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.data import clean_text, describe, find_tess, make_splits, save_json  # noqa: E402
from common.features import (load_audio, load_ssl, load_text_model, logmel, ssl_features,  # noqa: E402
                             text_features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tess", required=True)
    ap.add_argument("--out", default="features")
    ap.add_argument("--ssl", default="microsoft/wavlm-base-plus")
    ap.add_argument("--text_model", default="bert-base-uncased")
    ap.add_argument("--snrs", type=int, nargs="*", default=[20, 10, 0])
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(a.out, exist_ok=True)

    items = find_tess(a.tess)
    describe(items)
    assert items, f"no TESS wav files found under {a.tess}"
    splits = make_splits(items, a.split_seed)
    save_json(items, os.path.join(a.out, "metadata.json"))
    save_json(splits, os.path.join(a.out, "splits.json"))
    for k, v in splits.items():
        print(f"split {k}: " + ", ".join(f"{p} {len(ids)}" for p, ids in v.items()))

    t0 = time.time()
    fe, ssl = load_ssl(a.ssl, dev, a.smoke)
    wavlm, mel, lengths = {}, {}, []
    for n, x in enumerate(items, 1):
        wav = load_audio(x["path"])
        lengths.append(len(wav) / 16000)
        wavlm[x["id"]] = ssl_features(wav, fe, ssl, dev)
        mel[x["id"]] = torch.from_numpy(logmel(wav))
        if n % 500 == 0:
            print(f"speech {n}/{len(items)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"trimmed clip length: mean {sum(lengths) / len(lengths):.2f} s, min {min(lengths):.2f}, "
          f"max {max(lengths):.2f}")
    torch.save(wavlm, os.path.join(a.out, "speech_wavlm.pt"))
    torch.save(mel, os.path.join(a.out, "speech_logmel.pt"))
    del wavlm

    test_ids = set(splits["word"]["test"])
    for snr in a.snrs:
        noisy = {x["id"]: ssl_features(load_audio(x["path"], noise_snr=snr, seed=i), fe, ssl, dev)
                 for i, x in enumerate(items) if x["id"] in test_ids}
        torch.save(noisy, os.path.join(a.out, f"speech_wavlm_snr{snr}.pt"))
        print(f"noisy test set at {snr} dB: {len(noisy)} clips", flush=True)
    del ssl

    tok, bert = load_text_model(a.text_model, dev, a.smoke)
    text = {x["id"]: text_features(clean_text(x["text"]), tok, bert, dev) for x in items}
    torch.save(text, os.path.join(a.out, "text_bert.pt"))
    save_json({x["id"]: tok.convert_ids_to_tokens(tok(clean_text(x["text"]))["input_ids"]) for x in items},
              os.path.join(a.out, "text_tokens.json"))
    ex = items[0]
    print(f"text example: {ex['text']!r} -> {tok.tokenize(clean_text(ex['text']))}")
    print(f"features done in {time.time() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
