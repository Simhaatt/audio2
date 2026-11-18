"""Build the self-contained Kaggle notebooks.

    python kaggle/build_notebook.py                 -> kaggle/TESS_emotion.ipynb (the main experiment)
    python kaggle/build_notebook.py augmentation    -> kaggle/TESS_augmentation.ipynb (speaker augmentation)

The notebook embeds common/, models/ and scripts/ as a base64 tarball (CODE_BUILD stamps the build), finds the
TESS dataset under /kaggle/input, extracts features, trains and tests every variant, makes the tables and plots,
and zips Results/ to /kaggle/working/TESS_results.zip. The augmentation notebook runs the cross-speaker
speaker-augmentation experiment and zips Results_aug/ to /kaggle/working/TESS_aug_results.zip.
"""
import sys
import base64
import hashlib
import io
import json
import os
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def code_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in ("common", "models", "scripts"):
            for dirpath, _, files in os.walk(os.path.join(ROOT, d)):
                for f in sorted(files):
                    if f.endswith(".py"):
                        p = os.path.join(dirpath, f)
                        tar.add(p, arcname=os.path.relpath(p, ROOT).replace(os.sep, "/"))
    return base64.b64encode(buf.getvalue()).decode()


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


def augmentation_cells():
    variants = "speech_mean speech_mean_aug speech_logmel speech_logmel_aug"
    return [
        cell("def run(cmd):\n"
             "    print('+', cmd, flush=True)\n"
             "    t0 = time.time()\n"
             "    p = subprocess.run(cmd, shell=True, cwd=PROJ)\n"
             "    print(f'({time.time() - t0:.0f} s)')\n"
             "    assert p.returncode == 0, f'failed: {cmd}'\n\n"
             "FEAT = '/tmp/tess_features_aug'\n"
             "run(f'python scripts/extract_augmented.py --tess /kaggle/input --out {FEAT}')"),
        cell(f"run(f'python scripts/run_all.py --features {{FEAT}} --results Results_aug --seeds 0 1 2 --epochs 40 "
             f"--splits oaf2yaf yaf2oaf --variants {variants}')"),
        cell("run('python scripts/make_results.py --results Results_aug')\n"
             "print(open(f'{PROJ}/Results_aug/accuracy_table.md').read())"),
        cell("import glob, zipfile\n"
             "keep = {f'Results_aug/runs/{v}/{s}/seed0/model.pt' for v in ('speech_mean', 'speech_mean_aug')\n"
             "        for s in ('oaf2yaf', 'yaf2oaf')}\n"
             "out = '/kaggle/working/TESS_aug_results.zip'\n"
             "with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:\n"
             "    for p in glob.glob(f'{PROJ}/Results_aug/**/*', recursive=True):\n"
             "        rel = os.path.relpath(p, PROJ).replace(os.sep, '/')\n"
             "        if os.path.isfile(p) and (not rel.endswith('model.pt') or rel in keep):\n"
             "            z.write(p, rel)\n"
             "    for f in ('metadata.json', 'splits.json'):\n"
             "        z.write(f'{FEAT}/{f}', f'Results_aug/data/{f}')\n"
             "print(out, round(os.path.getsize(out) / 1e6, 1), 'MB | code build', CODE_BUILD)"),
    ]


def main():
    b64 = code_b64()
    build = time.strftime("%Y-%m-%d %H:%M") + " " + hashlib.sha1(b64.encode()).hexdigest()[:8]
    cells = [
        cell("# Multimodal emotion recognition on TESS (speech, text, fusion)\n\n"
             "1. **Add data:** the dataset *Toronto emotional speech set (TESS)* "
             "(`ejlok1/toronto-emotional-speech-set-tess`).\n"
             "2. **Settings:** Accelerator *GPU T4 x1* (or P100), Internet *on* (downloads WavLM and BERT).\n"
             "3. **Save Version > Save & Run All (Commit).** About 1.5-2 hours.\n"
             "4. Download `TESS_results.zip` from the Output tab.", "markdown"),
        cell(f"CODE_B64 = '{b64}'\nCODE_BUILD = '{build}'"),
        cell("import base64, io, os, sys, tarfile, subprocess, time, torch\n"
             "PROJ = '/kaggle/working/project'\n"
             "os.makedirs(PROJ, exist_ok=True)\n"
             "tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode='r:gz').extractall(PROJ)\n"
             "print('code build:', CODE_BUILD)\n"
             "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (enable a GPU!)')\n"
             "import transformers, librosa, sklearn\n"
             "print('torch', torch.__version__, '| transformers', transformers.__version__, '| librosa',\n"
             "      librosa.__version__, '| sklearn', sklearn.__version__)\n"
             "wavs = [os.path.join(d, f) for d, _, fs in os.walk('/kaggle/input') for f in fs if f.lower().endswith('.wav')]\n"
             "print(len(wavs), 'wav files under /kaggle/input, e.g.', wavs[:2])\n"
             "assert wavs, 'TESS not found: add the dataset (Add data -> ejlok1/toronto-emotional-speech-set-tess)'"),
        cell("def run(cmd):\n"
             "    print('+', cmd, flush=True)\n"
             "    t0 = time.time()\n"
             "    p = subprocess.run(cmd, shell=True, cwd=PROJ)\n"
             "    print(f'({time.time() - t0:.0f} s)')\n"
             "    assert p.returncode == 0, f'failed: {cmd}'\n\n"
             "FEAT = '/tmp/tess_features'   # about 7 GB, not part of the output\n"
             "run(f'python scripts/extract_features.py --tess /kaggle/input --out {FEAT}')"),
        cell("run(f'python scripts/run_all.py --features {FEAT} --results Results --seeds 0 1 2 --epochs 40')"),
        cell("run(f'python scripts/text_zero_shot.py --features {FEAT} --results Results')"),
        cell("run('python scripts/make_results.py --results Results')\n"
             "print(open(f'{PROJ}/Results/accuracy_table.md').read())"),
        cell("# keep weights of seed 0 for the three main models; every run keeps metrics, predictions, reps, logs\n"
             "import glob, zipfile\n"
             "keep = {f'Results/runs/{v}/{s}/seed0/model.pt'\n"
             "        for v in ('speech', 'speech_spknorm', 'text', 'fusion', 'fusion_xattn')\n"
             "        for s in ('word', 'oaf2yaf', 'yaf2oaf')}\n"
             "out = '/kaggle/working/TESS_results.zip'\n"
             "with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:\n"
             "    for p in glob.glob(f'{PROJ}/Results/**/*', recursive=True):\n"
             "        rel = os.path.relpath(p, PROJ).replace(os.sep, '/')\n"
             "        if os.path.isfile(p) and (not rel.endswith('model.pt') or rel in keep):\n"
             "            z.write(p, rel)\n"
             "    for f in ('metadata.json', 'splits.json', 'text_tokens.json'):\n"
             "        z.write(f'{FEAT}/{f}', f'Results/data/{f}')\n"
             "print(out, round(os.path.getsize(out) / 1e6, 1), 'MB | code build', CODE_BUILD)"),
    ]
    if len(sys.argv) > 1 and sys.argv[1] == "augmentation":
        cells = cells[1:3] + augmentation_cells()
        cells.insert(0, cell("# Speaker augmentation on TESS (cross-speaker splits)\n\n"
                             "Same setup as `TESS_emotion.ipynb`: add the TESS dataset "
                             "(`ejlok1/toronto-emotional-speech-set-tess`), GPU T4 x1, Internet on, "
                             "Save & Run All. About 1 hour. Download `TESS_aug_results.zip` from the Output tab.",
                             "markdown"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                       "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    name = "TESS_augmentation.ipynb" if len(sys.argv) > 1 and sys.argv[1] == "augmentation" else "TESS_emotion.ipynb"
    path = os.path.join(ROOT, "kaggle", name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"{path} | code build {build} | {len(b64) / 1e3:.0f} kB of code")


if __name__ == "__main__":
    main()
