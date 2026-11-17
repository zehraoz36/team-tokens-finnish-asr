# Team Tokens – Finnish ASR

COE025 Natural Language Processing, Project 3.
Automatic speech recognition (ASR) for Finnish: zero-shot baselines vs. fine-tuned models.

## Team
Ece Mina Örenler, Tuana Harmankaya, Begüm Ziyade, Yunus Emre Demirel, Zehra Özcan

## Language
Finnish (fi). Not English and not a mother tongue of any group member.

## Data
| Dataset | Use | License |
|---|---|---|
| Mozilla Common Voice (fi) | train / dev / test | CC0 |
| Aalto Finnish Parliament ASR Corpus (dev/eval) | dev / test | CC BY 4.0 |
| Google FLEURS (fi_fi) | test | CC BY 4.0 |

Audio files are not stored in this repository. See `scripts/` for download and preparation.

## Repository structure
- `data/` – prepared train/dev/test lists (CSV)
- `scripts/` – data preparation, evaluation, fine-tuning
- `results/` – WER/CER tables and predictions

## Evaluation
WER and CER (jiwer), same test clips and the same text normalization for every model.
