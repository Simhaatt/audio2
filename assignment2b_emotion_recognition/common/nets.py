"""The trainable blocks. Pretrained encoders (WavLM, BERT) stay frozen; their cached features enter here.

LayerMix        softmax-weighted sum of the encoder's layers (learns which layers to use)
AttnPool        attention pooling: a learned score per time step / token, softmax over the valid ones
SpeechEncoder   temporal modelling: projection -> 2-layer BiLSTM -> attention pooling -> 256-d vector
TextEncoder     contextual modelling: projection -> 1 Transformer encoder layer -> attention pooling -> 256-d
GMU             gated multimodal unit (Arevalo et al., 2017): z = sigmoid(W[s; t]),
                h = z * tanh(W_s s) + (1 - z) * tanh(W_t t). z near 1 means the model relies on speech.
xattn fusion    cross-modal attention: every text token (query) attends over the speech frames (keys, values);
                the attended speech joins the token (residual + LayerNorm) and attention pooling over tokens
                gives the fused vector. The text cannot say which emotion it is, but it can tell the model where
                in the audio to listen; the attention maps show where each token looked.
Classifier      256 -> 128 -> 7 with ReLU and dropout

Each model returns (logits, reps); reps holds the block outputs used for the cluster plots.
"""
import torch
import torch.nn as nn

D = 256


class LayerMix(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x):  # (B, L, T, F) -> (B, T, F)
        return (torch.softmax(self.w, 0)[None, :, None, None] * x).sum(1)


class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, h, mask):  # h (B, T, D), mask (B, T) True = valid
        a = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, -1)
        return (a.unsqueeze(-1) * h).sum(1), a


class SpeechEncoder(nn.Module):
    def __init__(self, in_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.mix = LayerMix(n_layers) if n_layers > 1 else None
        self.proj = nn.Sequential(nn.Linear(in_dim, D), nn.LayerNorm(D), nn.Dropout(dropout))
        self.rnn = nn.LSTM(D, D // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttnPool(D)

    def forward(self, x, mask, return_seq=False):
        if self.mix is not None:
            x = self.mix(x)
        h = self.proj(x.float())
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=mask.shape[1])
        pooled = self.pool(h, mask)[0]
        return (pooled, h) if return_seq else pooled


class TextEncoder(nn.Module):
    def __init__(self, in_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.mix = LayerMix(n_layers) if n_layers > 1 else None
        self.proj = nn.Sequential(nn.Linear(in_dim, D), nn.LayerNorm(D), nn.Dropout(dropout))
        self.ctx = nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=2 * D, dropout=dropout, batch_first=True)
        self.pool = AttnPool(D)

    def forward(self, x, mask, return_seq=False):
        if self.mix is not None:
            x = self.mix(x)
        h = self.ctx(self.proj(x.float()), src_key_padding_mask=~mask)
        pooled = self.pool(h, mask)[0]
        return (pooled, h) if return_seq else pooled


class Classifier(nn.Module):
    def __init__(self, n_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, h):
        return self.net(h)


class SpeechNet(nn.Module):
    def __init__(self, in_dim, n_layers):
        super().__init__()
        self.enc = SpeechEncoder(in_dim, n_layers)
        self.head = Classifier()

    def forward(self, batch):
        h = self.enc(batch["speech"], batch["speech_mask"])
        return self.head(h), {"temporal": h}


class TextNet(nn.Module):
    def __init__(self, in_dim, n_layers):
        super().__init__()
        self.enc = TextEncoder(in_dim, n_layers)
        self.head = Classifier()

    def forward(self, batch):
        h = self.enc(batch["text"], batch["text_mask"])
        return self.head(h), {"contextual": h}


class GMU(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Linear(D, D)
        self.t = nn.Linear(D, D)
        self.z = nn.Linear(2 * D, D)

    def forward(self, s, t):
        z = torch.sigmoid(self.z(torch.cat([s, t], -1)))
        return z * torch.tanh(self.s(s)) + (1 - z) * torch.tanh(self.t(t)), z


class FusionNet(nn.Module):
    def __init__(self, speech_dim, speech_layers, text_dim, text_layers, fusion="gmu"):
        super().__init__()
        self.speech = SpeechEncoder(speech_dim, speech_layers)
        self.text = TextEncoder(text_dim, text_layers)
        self.fusion = fusion
        self.gmu = GMU() if fusion == "gmu" else None
        self.concat = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU()) if fusion == "concat" else None
        if fusion == "xattn":
            self.xattn = nn.MultiheadAttention(D, num_heads=4, dropout=0.1, batch_first=True)
            self.xnorm = nn.LayerNorm(D)
            self.xpool = AttnPool(D)
        self.head = Classifier()
        self.last_attention = None  # (B, tokens, frames) of the last batch, xattn only

    def forward(self, batch):
        if self.fusion == "xattn":
            s, s_seq = self.speech(batch["speech"], batch["speech_mask"], return_seq=True)
            t, t_seq = self.text(batch["text"], batch["text_mask"], return_seq=True)
            att, w = self.xattn(t_seq, s_seq, s_seq, key_padding_mask=~batch["speech_mask"], need_weights=True,
                                average_attn_weights=True)
            self.last_attention = w.detach()
            h = self.xpool(self.xnorm(att + t_seq), batch["text_mask"])[0]
            return self.head(h), {"temporal": s, "contextual": t, "fusion": h}
        s = self.speech(batch["speech"], batch["speech_mask"])
        t = self.text(batch["text"], batch["text_mask"])
        if self.gmu is not None:
            h, z = self.gmu(s, t)
            reps = {"temporal": s, "contextual": t, "fusion": h, "gate": z.mean(-1, keepdim=True)}
        else:
            h = self.concat(torch.cat([s, t], -1))
            reps = {"temporal": s, "contextual": t, "fusion": h}
        return self.head(h), reps
