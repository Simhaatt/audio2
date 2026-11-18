"""Shared training and evaluation code for the three pipelines.

Cached features (from scripts/extract_features.py) live in <features>/ as {id: tensor} dicts:
  speech_wavlm.pt (13 x T x 768), speech_logmel.pt (T x 64), text_bert.pt (13 x N x 768),
  speech_wavlm_snr<k>.pt (word-split test clips with white noise at k dB SNR).
Training: AdamW (lr 1e-3, weight decay 1e-2), batch 32, up to 40 epochs, cross-entropy with label smoothing 0.05;
the epoch with the best validation macro-F1 is kept (patience 8). Only the small heads train; the encoders stay frozen.
"""
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from common.data import EMOTIONS, load_json
from common.nets import FusionNet, SpeechNet, TextNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_CACHE = {}


def load_features(features_dir, name):
    """Cached {id: tensor}. A name ending in _spknorm is the base features normalised per speaker; a name ending
    in +aug adds the augmented copies (<name>_aug.pt, keys '<clip id>|<augmentation>')."""
    key = (features_dir, name)
    if key not in _CACHE:
        if name.endswith("+aug"):
            base = name[:-len("+aug")]
            _CACHE[key] = {**load_features(features_dir, base),
                           **torch.load(os.path.join(features_dir, base + "_aug.pt"))}
        elif name.endswith("_spknorm"):
            _CACHE[key] = speaker_normalise(load_features(features_dir, name[:-len("_spknorm")]),
                                            load_json(os.path.join(features_dir, "metadata.json")))
        else:
            _CACHE[key] = torch.load(os.path.join(features_dir, name + ".pt"))
    return _CACHE[key]


def speaker_normalise(feats, meta):
    """Standardise each feature dimension (per layer) with the mean and SD of that speaker's frames.

    The statistics need no labels, so they can be computed for a new speaker from its unlabelled recordings
    (here: all of that speaker's clips). This removes the speaker's average voice (pitch range, timbre,
    recording level) and leaves how each clip departs from it, which is where emotion shows."""
    spk = {m["id"]: m["speaker"] for m in meta}
    stats = {}
    for i, x in feats.items():
        x = x.double()
        s, ss, n = stats.get(spk[i], (0, 0, 0))
        stats[spk[i]] = (s + x.sum(-2), ss + (x ** 2).sum(-2), n + x.shape[-2])
    norm = {}
    for k, (s, ss, n) in stats.items():
        mean = s / n
        norm[k] = (mean.float(), (ss / n - mean ** 2).clamp(min=1e-8).sqrt().float())
    return {i: ((x.float() - norm[spk[i]][0].unsqueeze(-2)) / norm[spk[i]][1].unsqueeze(-2)).half()
            for i, x in feats.items()}


def _pad(seqs, time_axis):
    """Pad along time_axis; returns (stacked, mask)."""
    lens = [s.shape[time_axis] for s in seqs]
    T = max(lens)
    out = []
    for s in seqs:
        pad = [0, 0] * (s.dim() - time_axis - 1) + [0, T - s.shape[time_axis]]
        out.append(nn.functional.pad(s, pad))
    mask = torch.arange(T)[None, :] < torch.tensor(lens)[:, None]
    return torch.stack(out), mask


class Data:
    """Features + labels for one split, for the modalities a pipeline needs."""

    def __init__(self, features_dir, ids, speech=None, text=None):
        meta = {m["id"]: m for m in load_json(os.path.join(features_dir, "metadata.json"))}
        self.ids = list(ids)
        # augmented copies ('<clip id>|<augmentation>') share their original clip's label and metadata
        self.labels = torch.tensor([meta[i.split("|")[0]]["label"] for i in self.ids])
        self.meta = [meta[i.split("|")[0]] for i in self.ids]
        self.speech = load_features(features_dir, speech) if speech else None
        self.text = load_features(features_dir, text) if text else None
        # which clip each modality's features come from; shuffle_modality() mismatches them on purpose
        self.speech_ids, self.text_ids = list(self.ids), list(self.ids)

    def shuffle_modality(self, which, seed=0):
        """Give every clip the speech (or text) of another, random test clip: the labels stay, the cue goes."""
        perm = np.random.default_rng(seed).permutation(len(self.ids))
        setattr(self, f"{which}_ids", [self.ids[p] for p in perm])
        return self

    def batches(self, batch_size, shuffle, device):
        order = np.random.permutation(len(self.ids)) if shuffle else np.arange(len(self.ids))
        for k in range(0, len(order), batch_size):
            idx = order[k:k + batch_size]
            b = {"y": self.labels[idx].to(device), "idx": idx}
            if self.speech is not None:
                s = [self.speech[self.speech_ids[i]] for i in idx]
                b["speech"], b["speech_mask"] = _pad(s, 1 if s[0].dim() == 3 else 0)
            if self.text is not None:
                t = [self.text[self.text_ids[i]] for i in idx]
                b["text"], b["text_mask"] = _pad(t, 1 if t[0].dim() == 3 else 0)
            yield {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    def dims(self, which):
        x = next(iter((self.speech if which == "speech" else self.text).values()))
        return (x.shape[-1], x.shape[0] if x.dim() == 3 else 1)


def build_model(pipeline, train, fusion="gmu"):
    if pipeline == "speech":
        return SpeechNet(*train.dims("speech"))
    if pipeline == "text":
        return TextNet(*train.dims("text"))
    return FusionNet(*train.dims("speech"), *train.dims("text"), fusion=fusion)


@torch.no_grad()
def predict(model, data, device, batch_size=64):
    model.eval()
    logits, reps = [], {}
    for b in data.batches(batch_size, False, device):
        out, r = model(b)
        logits.append(out.float().cpu())
        for k, v in r.items():
            reps.setdefault(k, []).append(v.float().cpu())
    logits = torch.cat(logits)
    return logits, {k: torch.cat(v).numpy() for k, v in reps.items()}


def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "confusion": confusion_matrix(y, pred, labels=list(range(len(EMOTIONS)))).tolist(),
            "per_class_recall": {e: float(np.mean(np.array(pred)[np.array(y) == i] == i)) if np.any(np.array(y) == i)
                                 else None for i, e in enumerate(EMOTIONS)}}


def train(model, train_data, val_data, device, epochs=40, lr=1e-3, batch_size=32, patience=8, log=print):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in train_data.batches(batch_size, True, device):
            logits, _ = model(b)
            loss = loss_fn(logits, b["y"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b["y"])
            n += len(b["y"])
        vlogits, _ = predict(model, val_data, device)
        vloss = loss_fn(vlogits, val_data.labels).item()
        s = scores(val_data.labels.numpy(), vlogits.argmax(1).numpy())
        history.append({"epoch": ep, "train_loss": tot / n, "val_loss": vloss, "val_accuracy": s["accuracy"],
                        "val_macro_f1": s["macro_f1"]})
        log(f"epoch {ep:2d} train loss {tot / n:.4f} | val loss {vloss:.4f} acc {s['accuracy']:.4f} "
            f"f1 {s['macro_f1']:.4f}")
        if s["macro_f1"] > best + 1e-6:
            best, bad = s["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return history


def write_predictions(path, data, logits):
    prob = torch.softmax(logits, 1).numpy()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "speaker", "word", "true", "pred", "confidence"] + [f"p_{e}" for e in EMOTIONS])
        for m, p in zip(data.meta, prob):
            w.writerow([m["id"], m["speaker"], m["word"], m["emotion"], EMOTIONS[int(p.argmax())],
                        round(float(p.max()), 4)] + [round(float(x), 4) for x in p])


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
