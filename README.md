# Assignment 2: forced alignment (2a) and multimodal emotion recognition (2b)

| Part | Folder | Report |
|---|---|---|
| **2a** Forced alignment with the Montreal Forced Aligner | [`assignment2a_forced_alignment/`](assignment2a_forced_alignment) | [report.pdf](assignment2a_forced_alignment/report/report.pdf) |
| **2b** Multimodal emotion recognition on TESS (speech, text, fusion) | [`assignment2b_emotion_recognition/`](assignment2b_emotion_recognition) | [report.pdf](assignment2b_emotion_recognition/report/report.pdf) |

Each folder is self-contained, with its own README (setup and exact commands), `requirements.txt`, code, outputs and
report.

## 2a: Forced alignment (MFA 3.4.3)

Six provided recordings (three BU Radio News clips, three non-native ISLE minimal pairs) aligned at word and phone
level, with four setups compared: the `english_us_arpa` and `english_mfa` acoustic models, the pretrained
dictionaries extended by G2P pronunciations for out-of-vocabulary words, and a trailing-silence remedy. One command,
`python run_pipeline.py`, runs everything.

- The news files are NIST SPHERE audio with a `.wav` name; the pipeline converts them before MFA can read them.
- The two acoustic models agree on 90 % of word boundaries within 20 ms; `english_mfa` stretches word-final stops
  over the trailing silence in two clips.
- **The aligner as a pronunciation checker:** given both vowels of each minimal pair, the aligner hears the
  learner's "bad" as "bed" (the only one of six words whose partner vowel fits better), matching the formants.
- TextGrids: `assignment2a_forced_alignment/outputs/textgrids/`.

## 2b: Multimodal emotion recognition on TESS

Speech-only (frozen WavLM Base+ → BiLSTM + attention), text-only (frozen BERT → Transformer layer + attention) and
fusion (gated multimodal unit; concatenation and cross-modal attention as ablations), evaluated on unseen words and
on an unseen speaker, 3 seeds each.

| Model | Unseen words | OAF → YAF | YAF → OAF |
|---|---|---|---|
| Speech | 99.8 % | 65.4 % | 60.2 % |
| Text | 14.3 % | 14.3 % | 14.3 % |
| Fusion | 99.1 % | 59.5 % | 57.0 % |

- Every TESS sentence is "Say the word ___" in all seven emotions, so no text model can beat 1/7; the fusion
  models learn to ignore the text (shuffling it changes nothing, shuffling the speech drops them to chance).
- The cross-speaker errors are mislabelled clusters: one labelled clip per emotion of the new speaker lifts accuracy
  from 65 % to 92 %.
- Speaker augmentation (pitch and speed changes) adds 20 points to log-mel features but little to WavLM.
- Training ran on a Kaggle T4 GPU with the notebooks in `assignment2b_emotion_recognition/kaggle/`.
