"""Preprocessing and feature extraction for both modalities. Features are computed once and cached.

Speech preprocessing: load as mono 16 kHz (the rate WavLM was trained on; TESS is recorded at 24.4 kHz),
trim leading/trailing silence (librosa, 30 dB below the peak), scale the peak to 0.95, cap at 4 s.
Speech features (time_steps x features):
  wavlm   hidden states of every layer of a frozen WavLM Base+ (13 x T x 768, one frame per 20 ms).
          The model learns how to weight the layers (SUPERB-style), so it can use whichever layers carry
          the emotional cues (prosody, voice quality) best.
  logmel  64-band log-mel spectrogram, 25 ms window, 10 ms hop, normalised per utterance (T x 64).
          A classic hand-made baseline for the ablation.
  wavlmmean  the mean of WavLM's 13 layers (T x 768): 13 times smaller, used for the speaker-augmentation runs,
          where every training clip also exists in four augmented versions (see augment()).
Text preprocessing: lower case, punctuation removed, WordPiece tokens of BERT.
Text features (tokens x features): hidden states of every layer of a frozen BERT base (13 x N x 768).
"""
import numpy as np
import torch

SR = 16000
MAX_SEC = 4.0


AUGMENTATIONS = ["pitch-3", "pitch+3", "speed0.9", "speed1.1"]


def augment(wav, kind):
    """Speaker augmentation: make the clip sound like a different voice.
    pitch+-3   pitch shifted by 3 semitones at the same duration (librosa phase vocoder + resampling)
    speedX     resampled so that it plays X times faster: pitch, formants and tempo all scale by X, roughly what
               a shorter or longer vocal tract does (the 'speed perturbation' of Ko et al., 2015)"""
    import librosa
    if kind.startswith("pitch"):
        return librosa.effects.pitch_shift(wav, sr=SR, n_steps=float(kind[5:]))
    factor = float(kind[5:])
    return librosa.resample(wav, orig_sr=int(round(SR * factor)), target_sr=SR)


def load_audio(path, noise_snr=None, seed=0, augmentation=None):
    import librosa
    wav, _ = librosa.load(path, sr=SR, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    if augmentation is not None:
        wav = augment(wav, augmentation)
    wav = wav[: int(MAX_SEC * SR)]
    if noise_snr is not None:
        # white noise at a fixed SNR, the same noise for a given clip and seed
        rng = np.random.default_rng(seed)
        p_sig = np.mean(wav ** 2) + 1e-12
        wav = wav + rng.normal(0, np.sqrt(p_sig / 10 ** (noise_snr / 10)), size=wav.shape).astype(np.float32)
    return (0.95 * wav / (np.abs(wav).max() + 1e-9)).astype(np.float32)


def logmel(wav):
    import librosa
    m = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=400, hop_length=160, n_mels=64)
    m = np.log(m + 1e-6).T
    return ((m - m.mean(0)) / (m.std(0) + 1e-5)).astype(np.float16)


def load_ssl(name, device, smoke=False):
    from transformers import AutoFeatureExtractor, AutoModel
    if smoke:  # tiny random-weight model with the same interface, for testing without downloads
        from transformers import WavLMConfig, WavLMModel
        model = WavLMModel(WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                       intermediate_size=64, conv_dim=(32,) * 7))
        return None, model.to(device).eval()
    return AutoFeatureExtractor.from_pretrained(name), AutoModel.from_pretrained(name).to(device).eval()


@torch.no_grad()
def ssl_features(wav, fe, model, device):
    """(layers, frames, dim) float16"""
    x = fe(wav, sampling_rate=SR, return_tensors="pt").input_values if fe is not None else \
        torch.from_numpy((wav - wav.mean()) / (wav.std() + 1e-7))[None]
    out = model(x.to(device), output_hidden_states=True)
    return torch.stack(out.hidden_states, 0)[:, 0].half().cpu()


def load_text_model(name, device, smoke=False):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if smoke:
        from transformers import BertConfig, BertModel
        model = BertModel(BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2,
                                     num_attention_heads=2, intermediate_size=64))
        return tok, model.to(device).eval()
    return tok, AutoModel.from_pretrained(name).to(device).eval()


@torch.no_grad()
def text_features(text, tok, model, device):
    """(layers, tokens, dim) float16, tokens include [CLS] and [SEP]"""
    enc = tok(text, return_tensors="pt")
    out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
    return torch.stack(out.hidden_states, 0)[:, 0].half().cpu()
