"""Whole forced-alignment pipeline in one command.

    python run_pipeline.py                 # everything
    python run_pipeline.py --skip_align    # only re-draw figures and re-run the analysis

Steps
1. prepare   data/ -> outputs/corpus/<speaker>/<utt>.wav + .lab   (scripts/prepare_corpus.py)
2. download  pretrained MFA models if missing (acoustic, dictionary, G2P)
3. g2p       pronunciations for out-of-vocabulary words with MFA's G2P model -> outputs/dictionaries/*_oov.dict;
             custom dictionary = pretrained dictionary + those entries
4. align     one run per setup in SETUPS -> outputs/textgrids/<setup>/<speaker>/<utt>.TextGrid
5. check     pronunciation check of the ISLE minimal pairs (scripts/pronunciation_check.py)
6. analyse   figures and tables (scripts/visualize.py, scripts/analyze.py)

Needs `mfa` (Montreal Forced Aligner 3.x) on PATH and Python with numpy, scipy, matplotlib, praatio,
praat-parselmouth. Set MFA_TEMP to choose MFA's working folder (default: MFA_ROOT_DIR).
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
CORPUS = os.path.join(OUT, "corpus")
DICTS = os.path.join(OUT, "dictionaries")

TRIMMED = os.path.join(OUT, "corpus_trimmed")
# name: (acoustic model, pretrained dictionary, G2P model or None for "no G2P", corpus)
SETUPS = {
    "english_us_arpa_pretrained_dict": ("english_us_arpa", "english_us_arpa", None, CORPUS),
    "english_us_arpa": ("english_us_arpa", "english_us_arpa", "english_us_arpa", CORPUS),
    "english_mfa": ("english_mfa", "english_us_mfa", "english_us_mfa", CORPUS),
    # the fix for english_mfa's stretched final stops: trailing silence cut 25 dB below the peak
    "english_mfa_trimmed": ("english_mfa", "english_us_mfa", "english_us_mfa", TRIMMED),
}


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def mfa(*args):
    extra = ["--temporary_directory", os.environ["MFA_TEMP"]] if os.environ.get("MFA_TEMP") else []
    run(["mfa", *args, *extra])


def mfa_root():
    return os.environ.get("MFA_ROOT_DIR", os.path.join(os.path.expanduser("~"), "Documents", "MFA"))


def model_path(kind, name):
    ext = {"acoustic": ".zip", "dictionary": ".dict", "g2p": ".zip"}[kind]
    return os.path.join(mfa_root(), "pretrained_models", kind, name + ext)


def ensure_models():
    for acoustic, dictionary, g2p, _ in SETUPS.values():
        for kind, name in (("acoustic", acoustic), ("dictionary", dictionary), ("g2p", g2p)):
            if name and not os.path.exists(model_path(kind, name)):
                run(["mfa", "model", "download", kind, name])


def custom_dictionary(dictionary, g2p):
    """Pretrained dictionary + G2P pronunciations for the corpus's OOV words."""
    os.makedirs(DICTS, exist_ok=True)
    oov = os.path.join(DICTS, f"{dictionary}_oov.dict")
    mfa("g2p", CORPUS, g2p, oov, "--dictionary_path", dictionary, "--clean")
    full = os.path.join(DICTS, f"{dictionary}_custom.dict")
    with open(full, "w", encoding="utf-8") as out:
        out.write(open(model_path("dictionary", dictionary), encoding="utf-8").read().rstrip("\n") + "\n")
        out.write(open(oov, encoding="utf-8").read())
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_align", action="store_true")
    ap.add_argument("--setups", nargs="*", default=list(SETUPS))
    args = ap.parse_args()
    py = sys.executable
    if not args.skip_align:
        for corpus, extra in ((CORPUS, []), (TRIMMED, ["--trim_db", "25"])):
            shutil.rmtree(corpus, ignore_errors=True)
            run([py, os.path.join(ROOT, "scripts", "prepare_corpus.py"), os.path.join(ROOT, "data"), corpus] + extra)
        ensure_models()
        for name in args.setups:
            acoustic, dictionary, g2p, corpus = SETUPS[name]
            dict_arg = custom_dictionary(dictionary, g2p) if g2p else dictionary
            out = os.path.join(OUT, "textgrids", name)
            shutil.rmtree(out, ignore_errors=True)
            mfa("align", corpus, dict_arg, acoustic, out, "--clean", "--output_format", "long_textgrid")
        run([py, os.path.join(ROOT, "scripts", "pronunciation_check.py"), OUT])
    run([py, os.path.join(ROOT, "scripts", "visualize.py"), OUT])
    run([py, os.path.join(ROOT, "scripts", "analyze.py"), OUT])


if __name__ == "__main__":
    main()
