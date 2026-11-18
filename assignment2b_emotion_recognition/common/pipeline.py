"""Command-line entry points shared by models/<pipeline>/train.py and test.py.

    train.py --features <dir> --split word|oaf2yaf|yaf2oaf --seed 0 --out <run dir>
             [--speech_features wavlm|logmel|wavlm_spknorm] [--fusion gmu|concat|xattn]
    test.py  --features <dir> --run <run dir>

A run directory holds model.pt (weights + settings), history.json and train_log.txt; test.py adds metrics.json,
predictions.csv (one row per test clip) and reps.npz (block outputs of every test clip, for the cluster plots).
For the word split, test.py also scores the speech and fusion models on the noisy copies of the test set.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import EMOTIONS, SPLITS, load_json  # noqa: E402
from common.train_utils import (Data, build_model, load_features, predict, save_json, scores, set_seed,  # noqa: E402
                                train, write_predictions)

MODALITIES = {"speech": ("speech", None), "text": (None, "text"), "fusion": ("speech", "text")}


def feature_names(pipeline, speech_features):
    s, t = MODALITIES[pipeline]
    return (f"speech_{speech_features}" if s else None), ("text_bert" if t else None)


def device_of(arg):
    return torch.device(arg or ("cuda" if torch.cuda.is_available() else "cpu"))


def train_main(pipeline, argv=None):
    ap = argparse.ArgumentParser(description=f"train the {pipeline} pipeline")
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", choices=SPLITS, default="word")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speech_features", choices=["wavlm", "logmel", "wavlm_spknorm", "wavlmmean"], default="wavlm")
    ap.add_argument("--fusion", choices=["gmu", "concat", "xattn"], default="gmu")
    ap.add_argument("--augment", action="store_true",
                    help="speech only: add the augmented copies of every training clip (<features>_aug.pt)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet", action="store_true", help="log epochs to train_log.txt only")
    a = ap.parse_args(argv)
    set_seed(a.seed)
    dev = device_of(a.device)
    split = load_json(os.path.join(a.features, "splits.json"))[a.split]
    sname, tname = feature_names(pipeline, a.speech_features)
    train_ids = split["train"]
    if a.augment:
        assert pipeline == "speech", "augmentation is implemented for the speech pipeline"
        aug = load_features(a.features, sname + "+aug")
        keep = set(train_ids)
        train_ids = train_ids + sorted(k for k in aug if "|" in k and k.split("|")[0] in keep)
        tr = Data(a.features, train_ids, sname + "+aug", tname)
    else:
        tr = Data(a.features, train_ids, sname, tname)
    va = Data(a.features, split["val"], sname, tname)
    model = build_model(pipeline, tr, a.fusion)
    os.makedirs(a.out, exist_ok=True)
    log_f = open(os.path.join(a.out, "train_log.txt"), "w", encoding="utf-8")

    def log(msg):
        if not a.quiet:
            print(msg, flush=True)
        log_f.write(msg + "\n")

    n_par = sum(p.numel() for p in model.parameters())
    log(f"{pipeline} | split {a.split} | seed {a.seed} | train {len(tr.ids)} val {len(va.ids)} | "
        f"{n_par / 1e6:.2f} M trainable parameters | {dev}")
    history = train(model, tr, va, dev, epochs=a.epochs, log=log)
    torch.save({"state_dict": model.state_dict(), "pipeline": pipeline, "split": a.split, "seed": a.seed,
                "speech_features": a.speech_features, "fusion": a.fusion, "augment": a.augment},
               os.path.join(a.out, "model.pt"))
    save_json(history, os.path.join(a.out, "history.json"))
    log_f.close()


def test_main(pipeline, argv=None):
    ap = argparse.ArgumentParser(description=f"test the {pipeline} pipeline")
    ap.add_argument("--features", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    dev = device_of(a.device)
    ck = torch.load(os.path.join(a.run, "model.pt"), map_location="cpu")
    split = load_json(os.path.join(a.features, "splits.json"))[ck["split"]]
    sname, tname = feature_names(pipeline, ck["speech_features"])
    te = Data(a.features, split["test"], sname, tname)
    model = build_model(pipeline, te, ck["fusion"])
    model.load_state_dict(ck["state_dict"])
    model.to(dev)
    logits, reps = predict(model, te, dev)
    res = scores(te.labels.numpy(), logits.argmax(1).numpy())
    res.update({"pipeline": pipeline, "split": ck["split"], "seed": ck["seed"], "n_test": len(te.ids),
                "speech_features": ck["speech_features"], "fusion": ck["fusion"], "augment": ck.get("augment", False)})
    if pipeline == "fusion":  # how much the fused model relies on each modality
        res["modality_ablation"] = {}
        for which in ("speech", "text"):
            sh = Data(a.features, split["test"], sname, tname).shuffle_modality(which)
            sl, _ = predict(model, sh, dev)
            res["modality_ablation"][f"{which}_shuffled"] = scores(sh.labels.numpy(), sl.argmax(1).numpy())["accuracy"]
    if "gate" in reps:  # GMU: share of the fused vector taken from speech, per true emotion
        g = reps["gate"][:, 0]
        res["gate_speech_share"] = {"mean": float(g.mean()),
                                    **{e: float(g[te.labels.numpy() == i].mean()) for i, e in enumerate(EMOTIONS)}}
    if model.__class__.__name__ == "SpeechNet" or pipeline == "fusion":
        res["layer_weights"] = layer_weights(model)
    # noisy copies of the test speech (word split, WavLM features only)
    if sname == "speech_wavlm" and ck["split"] == "word":
        noise = {}
        for path in sorted(glob.glob(os.path.join(a.features, "speech_wavlm_snr*.pt"))):
            snr = re.search(r"snr(-?\d+)", path).group(1)
            nd = Data(a.features, split["test"], os.path.basename(path)[:-3], tname)
            nl, _ = predict(model, nd, dev)
            noise[snr] = {k: v for k, v in scores(nd.labels.numpy(), nl.argmax(1).numpy()).items()
                          if k in ("accuracy", "macro_f1")}
        res["noise"] = noise
    if ck["fusion"] == "xattn" and pipeline == "fusion":
        res["attention_time_centre"] = attention_summary(model, te, dev, a.features, a.run)
    save_json(res, os.path.join(a.run, "metrics.json"))
    write_predictions(os.path.join(a.run, "predictions.csv"), te, logits)
    np.savez_compressed(os.path.join(a.run, "reps.npz"), labels=te.labels.numpy(),
                        speakers=np.array([m["speaker"] for m in te.meta]), **reps)
    name = pipeline + (f" ({ck['speech_features']}{', augmented' if ck.get('augment') else ''})"
                       if pipeline == "speech" else "") + \
        (f" ({ck['fusion']})" if pipeline == "fusion" else "")
    print(f"{name} {ck['split']} seed {ck['seed']}: accuracy {res['accuracy']:.4f} macro-F1 {res['macro_f1']:.4f}"
          + (f" | shuffled {res['modality_ablation']}" if res.get("modality_ablation") else "")
          + (f" | noise {res['noise']}" if res.get("noise") else ""), flush=True)


@torch.no_grad()
def attention_summary(model, data, dev, features_dir, run, n_maps=12, frame_s=0.02):
    """Where in the clip each text token looked (cross-modal attention).

    For every test clip and token: the attention-weighted mean time of the speech frames, as a fraction of the
    clip (0 = start, 1 = end). Returns the mean per token position ([CLS], say, the, word, target word...), and
    writes attention.npz with full maps of the first n_maps clips for plotting."""
    tokens = load_json(os.path.join(features_dir, "text_tokens.json"))
    model.eval()
    centres, maps = {}, {}
    for b in data.batches(64, False, dev):
        model(b)
        w = model.last_attention.float().cpu()
        for j, i in enumerate(b["idx"]):
            n_tok, n_fr = int(b["text_mask"][j].sum()), int(b["speech_mask"][j].sum())
            a = w[j, :n_tok, :n_fr]
            t = (torch.arange(n_fr) + 0.5) / n_fr
            uid = data.ids[i]
            centres[uid] = ((a * t).sum(1) / a.sum(1).clamp(min=1e-9)).tolist()
            if len(maps) < n_maps:
                maps[uid] = a.numpy()
    # position names: the carrier phrase is the same in every sentence, the target word may span several tokens
    by_pos = {}
    for uid, c in centres.items():
        tk = tokens[uid]
        for k, v in enumerate(c):
            name = tk[k] if k <= 3 or k == len(tk) - 1 else "target word"
            by_pos.setdefault(name if name in ("[CLS]", "say", "the", "word", "[SEP]") else "target word", []).append(v)
    np.savez_compressed(os.path.join(run, "attention.npz"), ids=np.array(list(maps)),
                        **{f"map_{k}": v for k, v in maps.items()})
    save_json({k: tokens[k] for k in maps}, os.path.join(run, "attention_tokens.json"))
    return {k: float(np.mean(v)) for k, v in by_pos.items()}


def layer_weights(model):
    out = {}
    for name, m in model.named_modules():
        if m.__class__.__name__ == "LayerMix":
            out[name] = torch.softmax(m.w.detach().cpu(), 0).tolist()
    return out
