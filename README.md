# Team Tokens – Finnish ASR

COE025 Natural Language Processing, Project 3.
Automatic speech recognition (ASR) for Finnish: zero-shot baselines vs. fine-tuned models.

## Team
Ece Mina Örenler, Tuana Harmankaya, Begüm Ziyade, Yunus Emre Demirel, Zehra Özcan

## Language
Finnish (fi). Not English and not a mother tongue of any group member.

## Data
Prepared audio (16 kHz mono WAV): https://huggingface.co/datasets/zehraoz36/team-tokens-finnish-asr-data

```python
from datasets import load_dataset
ds = load_dataset("zehraoz36/team-tokens-finnish-asr-data")
```

| Split | Source | Clips | Hours | Speakers |
|---|---|---:|---:|---:|
| train | Common Voice 27.0 fi train + validated (dev/test speakers and sentences excluded) | 3770 | 4.65 | 10 |
| dev_cv | Common Voice dev | 1827 | 2.32 | 27 |
| dev_parl | Aalto Finnish Parliament dev | 1774 | 5.16 | 19 |
| test_cv | Common Voice test | 1833 | 2.61 | 187 |
| test_parl | Aalto Finnish Parliament eval | 1913 | 5.55 | 21 |
| test_fleurs | FLEURS fi_fi test | 916 | 3.29 | – |

Preparation: `scripts/prepare_data.ipynb` (download, 16 kHz conversion, text normalization, 0.5–30 s filter, split).
Clip lists without audio: `data/*.csv`.
Audio is not stored on GitHub because of its size (~3.5 GB).

### Sources and licenses
- Mozilla Common Voice Scripted Speech 27.0 – Finnish (CC0-1.0)
- Aalto Finnish Parliament ASR Corpus, dev/eval (Virkkunen, Mansikkaniemi & Kurimo; Zenodo 4581941; CC BY 4.0)
- Google FLEURS (CC BY 4.0)

## Evaluation
WER and CER (jiwer). Every model is scored on the same test clips with the same text normalization
(`normalize()` in `scripts/prepare_data.ipynb`). The three test sets are reported separately.

## Repository structure
- `data/` – clip lists (CSV)
- `scripts/` – data preparation, evaluation, fine-tuning
- `results/` – WER/CER tables and predictions
