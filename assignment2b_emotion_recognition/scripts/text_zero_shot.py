"""Can any text model recognise the emotion in TESS? Two checks that do not depend on our own text model.

1. Upper bound for every text-only classifier. A text model sees only the sentence, so it must give every clip of
   the same sentence the same answer. Its best possible accuracy is therefore sum over sentences of the most
   frequent label of that sentence, divided by the number of clips. In TESS each sentence occurs once per emotion
   and speaker, so the bound is 1/7.
2. A pretrained text-emotion classifier (j-hartmann/emotion-english-distilroberta-base: anger, disgust, fear,
   joy, neutral, sadness, surprise, trained on six English emotion datasets) applied zero-shot to every sentence.

    python scripts/text_zero_shot.py --features features --results Results
Writes Results/text_zero_shot.json and Results/text_zero_shot_words.csv.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.data import EMOTIONS, load_json  # noqa: E402

MODEL = "j-hartmann/emotion-english-distilroberta-base"
TO_TESS = {"anger": "angry", "disgust": "disgust", "fear": "fear", "joy": "happy", "neutral": "neutral",
           "sadness": "sad", "surprise": "ps"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--results", default="Results")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    meta = load_json(os.path.join(a.features, "metadata.json"))
    by_text = {}
    for m in meta:
        by_text.setdefault(m["text"], []).append(m["emotion"])
    bound = sum(max(Counter(v).values()) for v in by_text.values()) / len(meta)

    from transformers import pipeline
    clf = pipeline("text-classification", model=a.model, top_k=None)
    texts = sorted(by_text)
    preds = {}
    for t, out in zip(texts, clf(texts, batch_size=64)):
        best = max(out, key=lambda d: d["score"])
        preds[t] = (TO_TESS[best["label"]], best["score"])
    correct = sum(preds[m["text"]][0] == m["emotion"] for m in meta)
    pred_dist = Counter(p for p, _ in preds.values())
    res = {"clips": len(meta), "sentences": len(texts), "upper_bound_any_text_model": bound,
           "zero_shot_model": a.model, "zero_shot_accuracy": correct / len(meta),
           "zero_shot_predicted_label_share": {e: pred_dist.get(e, 0) / len(texts) for e in EMOTIONS},
           "most_confident_non_neutral": sorted(([t, p, round(s, 3)] for t, (p, s) in preds.items()
                                                 if p != "neutral"), key=lambda r: -r[2])[:10]}
    os.makedirs(a.results, exist_ok=True)
    json.dump(res, open(os.path.join(a.results, "text_zero_shot.json"), "w"), indent=1)
    with open(os.path.join(a.results, "text_zero_shot_words.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "predicted", "score", "true_labels_of_its_clips"])
        for t in texts:
            w.writerow([t, preds[t][0], round(preds[t][1], 3), " ".join(sorted(Counter(by_text[t])))])
    print(json.dumps({k: v for k, v in res.items() if k != "most_confident_non_neutral"}, indent=1))


if __name__ == "__main__":
    main()
