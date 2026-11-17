# Zehra Özcan

- Student ID: 2309011051
- GitHub: [@zehraoz36](https://github.com/zehraoz36)
- Repository: [zehraoz36/team-tokens-finnish-asr](https://github.com/zehraoz36/team-tokens-finnish-asr)
- Dataset: [zehraoz36/team-tokens-finnish-asr-data](https://huggingface.co/datasets/zehraoz36/team-tokens-finnish-asr-data)

## Summary

I set up the project repository and prepared the data used by every experiment: I downloaded three Finnish speech sources, converted all audio to one format, normalized the transcripts, removed unusable clips, built speaker-disjoint train/dev/test splits, and published the result as one Hugging Face dataset. The final data has **4.65 h of training speech**, **two dev sets (7.48 h)** and **three separate test sets (11.45 h)**, 23.6 h in total.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Project repository: structure, README, `.gitignore`, license | commits `5266f6b`, `a3286ff`, `2285cdd`, `4d953e9` |
| 2 | Dataset research and selection for Finnish | this file, section 1 |
| 3 | Data preparation notebook (download, conversion, normalization, filtering, splits) | `scripts/prepare_data.ipynb` (`bfdacb9`) |
| 4 | Clip lists for every split | `data/*.csv` (`dc03904`) |
| 5 | Prepared audio dataset on Hugging Face | [team-tokens-finnish-asr-data](https://huggingface.co/datasets/zehraoz36/team-tokens-finnish-asr-data) |
| 6 | Language tracker entry updated with the real data and hours | class tracker, row "Tokens" |

## 1. Data sources

| Source | License | Used for | Why |
|---|---|---|---|
| Mozilla Common Voice Scripted Speech 27.0 – Finnish | CC0-1.0 | train, dev, test | Read speech, many speakers, official speaker-disjoint splits |
| Aalto Finnish Parliament ASR Corpus, dev/eval (Zenodo 4581941) | CC BY 4.0 | dev, test | Parliament speech; dev/eval transcripts are **hand-corrected** (trusted references) |
| Google FLEURS fi_fi | CC BY 4.0 | test | Standard multilingual benchmark |

Also checked and not used: Lahjoita puhetta / Donate Speech (needs an approved research application, not possible before the deadline), CSS10 Finnish (single speaker), VoxPopuli fi (only ~6 h), YODAS (partly automatic subtitles, not trusted references).

## 2. Preparation steps (`scripts/prepare_data.ipynb`)

1. **Audio:** every clip converted with ffmpeg to 16 kHz, mono, 16-bit WAV (Common Voice is 48 kHz MP3; the other sources also differ).
2. **Text normalization** (`normalize()`): Unicode NFKC, lowercase, remove tags such as `[noise]` and `<unk>`, join thousands (`1 500` → `1500`), punctuation → space, collapse spaces. Finnish letters å/ä/ö are kept. The same function must be applied to model predictions before WER/CER.
3. **Parliament splits:** dev/eval membership taken from the official `seen_*.list` / `unseen_*.list` files; speaker taken from the file name.
4. **Filtering:** removed clips with unreadable audio, empty text, or duration outside 0.5–30 s (Whisper's 30 s window). Result: 0 bad audio, 0 empty text, 40 clips outside the length range, 0 duplicates.
5. **Larger training set:** the official Common Voice train split was only 2.55 h. I added clips from `validated.tsv` whose **speaker and sentence do not appear in any dev or test set** (+1,703 clips → 4.65 h).
6. **Leakage check:** no speaker overlap between Common Voice train and test (0 shared speakers).

## 3. Final data

| Split | Source | Clips | Hours | Speakers |
|---|---|---:|---:|---:|
| train | Common Voice train + validated | 3,770 | 4.65 | 10 |
| dev_cv | Common Voice dev | 1,827 | 2.32 | 27 |
| dev_parl | Parliament dev | 1,774 | 5.16 | 19 |
| test_cv | Common Voice test | 1,833 | 2.61 | 187 |
| test_parl | Parliament eval | 1,913 | 5.55 | 21 |
| test_fleurs | FLEURS test | 916 | 3.29 | – |

FLEURS does not provide speaker IDs.

## 4. Why the audio is on Hugging Face

The prepared audio is about 3.5 GB; GitHub does not accept files over 100 MB. The repository holds the code and the clip lists, and the audio is a public Hugging Face dataset that teammates load with one line:

    from datasets import load_dataset
    ds = load_dataset("zehraoz36/team-tokens-finnish-asr-data")

## 5. Limitations

- The training set is small (4.65 h) and has only 10 speakers; a few speakers recorded most Finnish Common Voice clips.
- Only the dev/eval part of the Parliament corpus is used; its 3,100 h training part is too large for our compute and time.
- Large public corpora (Common Voice, Parliament) may be in the pre-training data of some checkpoints; FLEURS is the safest comparison.

## Commit evidence

Repository [zehraoz36/team-tokens-finnish-asr](https://github.com/zehraoz36/team-tokens-finnish-asr), branch `main`:

- [`5266f6b`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/5266f6b) (2026-11-17) – initial commit (repository, license)
- [`a3286ff`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/a3286ff) (2026-11-17) – README with project details and structure
- [`2285cdd`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/2285cdd) (2026-11-17) – `.gitignore` for audio and raw data
- [`dc03904`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/dc03904) (2026-11-18) – prepared clip lists (`data/`, 7 files)
- [`bfdacb9`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/bfdacb9) (2026-11-18) – data preparation notebook
- [`4d953e9`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/4d953e9) (2026-11-18) – README with dataset table, sources and evaluation policy
