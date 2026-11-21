# Team Tokens — Finnish ASR

COE025 Natural Language Processing · Project 3.
Automatic speech recognition (ASR) for Finnish: pretrained baselines vs. a fine-tuned model,
evaluated across domains.

## Team

Ece Mina Örenler, Tuana Harmankaya, Begüm Ziyade, Yunus Emre Demirel, Zehra Özcan

## Language

Finnish (`fi`). Not English and not a mother tongue of any group member. Finnish is
agglutinative and morphologically rich, which makes it a demanding ASR target.

## Scope

We fine-tune only on Finnish read speech (Common Voice) and evaluate on three test sets that span
different domains: Common Voice (in-domain read speech), Aalto Parliament (out-of-domain
spontaneous/formal speech), and FLEURS (out-of-domain read speech). The large Aalto Parliament
training corpus (~3,100 h) is deliberately **not** used for training. This makes the study a
**cross-domain ASR robustness evaluation** — how models behave across domains — rather than a
single-domain adaptation. The small training set (4.65 h) is a known limitation and is part of
what this cross-domain setup tests.

## Data

Prepared audio (16 kHz mono WAV) is published as one Hugging Face dataset:

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

Total: 23.6 h (train 4.65 h, dev 7.48 h, test 11.45 h). Clip lists without audio: `data/*.csv`
(each row has `text` = cleaned/normalized and `raw_text` = original transcript). Per-split speaker,
utterance and hour counts, and the number of test sentences that also occur word for word in train,
are in `results/data_checks.md` (`scripts/data_checks.py`).

### Sources and licenses

| Source | License | Used for |
|---|---|---|
| Mozilla Common Voice Scripted Speech 27.0 – Finnish | CC0-1.0 | train, dev, test |
| Aalto Finnish Parliament ASR Corpus, dev/eval (Zenodo 4581941) | CC BY 4.0 | dev, test |
| Google FLEURS fi_fi | CC BY 4.0 | test |

### Splits (speaker-disjoint)

Splits are built **by speaker**, not by clip: all clips of a speaker stay in one split.
We verified **0 shared speakers** between train and every dev/test set, so no speaker leaks
from training into testing. FLEURS provides no speaker IDs. Preparation and the split logic:
`scripts/prepare_data.ipynb`.

## Hugging Face resources

Large files are hosted on Hugging Face, not GitHub, because GitHub rejects files over 100 MB.
This repository holds the code, clip lists, results and predictions; the audio and the fine-tuned
model live on Hugging Face and load with one line.

- **Dataset (prepared audio + transcripts):**
  https://huggingface.co/datasets/zehraoz36/team-tokens-finnish-asr-data
- **Fine-tuned model:**
  https://huggingface.co/tia88na/whisper-small-fi-tokens

  ```python
  from transformers import WhisperForConditionalGeneration, WhisperProcessor
  model = WhisperForConditionalGeneration.from_pretrained("tia88na/whisper-small-fi-tokens")
  processor = WhisperProcessor.from_pretrained("tia88na/whisper-small-fi-tokens")
  ```

## Evaluation

All systems use one shared evaluation module, `scripts/asr_eval.py`: references are the dataset's
`text` column, the project `normalize()` is applied once to each raw model output, WER/CER are
corpus-level (total edits / total reference words or characters), and RTF = compute time / audio
time. It writes `results/scores/<system>.csv` (exact checkpoint revision, WER/CER, substitutions/
deletions/insertions, RTF/RTFx, hardware, precision, settings) and per-clip predictions
`results/predictions/<system>__<test_set>.csv` (raw and normalized hypothesis). The three test
sets are reported **separately** and never averaged.

We report both WER and CER because Finnish is agglutinative: a single wrong suffix turns a whole
word into an error and inflates WER, while CER stays low. Metric choice follows the Hugging Face
Audio Course, Chapter 5, "Evaluation":
https://huggingface.co/learn/audio-course/en/chapter5/evaluation

95% bootstrap confidence intervals (cluster and clip resampling, with paired differences) are in
`results/bootstrap_ci.md` (`scripts/bootstrap_ci.py`).

## Baselines

Every pretrained model is run in standard form (no extra training by our group) and is a
**baseline**; our own fine-tuning is the **contribution**. We cover all method families from the
brief and report each system with its exact checkpoint and revision (`results/scores/`).

| Method family | System (checkpoint) | Finnish support | Status |
|---|---|---|---|
| Classical MFCC + GMM-HMM | `local/kaldi-grapheme-mono` (trained by us) | trained on our data | done |
| Self-supervised + CTC | `jonatasgrosman/wav2vec2-large-xlsr-53-finnish` | yes | done |
| Self-supervised + CTC (HuBERT) | — | no public Finnish HuBERT+CTC checkpoint | not run (see note) |
| Attention encoder-decoder | `openai/whisper-small` | yes (multilingual) | done |
| Attention encoder-decoder | `nvidia/canary-1b-v2` | yes | done |
| Transducer (TDT) | `nvidia/parakeet-tdt-0.6b-v3` | yes | done |
| Large audio-language model | `Qwen/Qwen3-ASR-1.7B` | yes | done |

**HuBERT + CTC note:** there is no public Finnish HuBERT model with an ASR/CTC head. Multilingual
HuBERT encoders such as `utter-project/mHuBERT-147` cover Finnish but ship no ASR head, and the
brief states a bare encoder is not a baseline. The self-supervised + CTC family is therefore
represented by the Finnish wav2vec 2.0 + CTC model above.

## Results

WER / CER in %, corpus-level, same test clips, references and `normalize()` for every system.
Full table with checkpoints, revisions, RTF and hardware: `results/results_table.md`
(`scripts/make_results_table.py`). Confidence intervals: `results/bootstrap_ci.md`.

### Baselines (pretrained, no training by our group)

| System | test_cv WER / CER | test_parl WER / CER | test_fleurs WER / CER |
|---|---:|---:|---:|
| parakeet-tdt-0.6b-v3 | **7.17** / 1.37 | 13.28 / 4.09 | 13.35 / 2.72 |
| canary-1b-v2 | 10.36 / 2.00 | **10.14** / 3.65 | **8.88** / 2.10 |
| qwen3-asr-1.7b zero-shot | 31.13 / 7.23 | 26.50 / 6.94 | 25.16 / 5.81 |
| whisper-small zero-shot | 30.94 / 6.71 | 25.68 / 8.40 | 26.37 / 5.26 |
| wav2vec2-xlsr-53-finnish + CTC | 42.78 / 8.93 | 56.22 / 13.07 | 59.52 / 15.83 |
| gmm-hmm monophone (trained by us) | 80.69 / 42.58 | 87.20 / 47.86 | 93.11 / 53.07 |

### Our contribution (trained by our group)

| System | test_cv WER / CER | test_parl WER / CER | test_fleurs WER / CER |
|---|---:|---:|---:|
| whisper-small fine-tuned (ours) | 23.93 / 4.61 | 24.72 / 5.89 | 28.85 / 5.97 |

Bold marks the lowest WER on that test set across all systems.

### Discussion

- **Strongest baselines:** `parakeet-tdt-0.6b-v3` and `canary-1b-v2` (large, recent NVIDIA models)
  lead on every test set by a wide margin. The classical GMM-HMM and the older wav2vec2 + CTC model
  are much weaker, as expected.
- **Our contribution:** fine-tuning `whisper-small` improves clearly over its own zero-shot baseline
  (test_cv WER 30.94 → 23.93, −7.0; test_parl CER 8.40 → 5.89) and beats the Qwen3-ASR baseline on
  the in-domain set. It does **not** reach Parakeet/Canary — they are far larger and trained on much
  more Finnish speech — which is the honest result for a 4.65 h fine-tune.
- **Cross-domain behaviour:** fine-tuning helps most in-domain (test_cv), is mixed on Parliament
  (WER barely changes, CER drops clearly) and regresses on the out-of-domain FLEURS read-speech set
  (WER +2.5). Specializing on the training domain reduces generalization, which is why we never
  average across test sets.
- **WER high, CER low** throughout, as expected for an agglutinative language where one wrong suffix
  fails a whole word.

## Repository structure

- `data/` — clip lists (CSV), one per split, plus `summary.csv` (`text` = cleaned, `raw_text` = original)
- `scripts/` — `prepare_data.ipynb` (data), `finnish_asr_whisper_finetune.ipynb` (fine-tuning),
  `asr_eval.py` (shared evaluation), `eval_*.ipynb` (one notebook per evaluated system),
  `make_results_table.py`, `bootstrap_ci.py`, `data_checks.py`
- `results/` — `scores/` (one file per system), `predictions/` (per-clip predictions),
  `run_metadata/` (per-system protocol and config), `results_table.md`, `bootstrap_ci.md`,
  `data_checks.md`, `legacy/` (first-run results)
- `contributions/` — per-member contribution notes

## Reproduce

1. `load_dataset("zehraoz36/team-tokens-finnish-asr-data")`
2. Fine-tuning: `scripts/finnish_asr_whisper_finetune.ipynb` in Colab (T4 GPU), top to bottom.
3. Score any system with the shared code via its `scripts/eval_<system>.ipynb`; each writes
   `results/scores/<system>.csv` and `results/predictions/<system>__<test_set>.csv`.
4. Regenerate the summaries: `python scripts/make_results_table.py`, `python scripts/bootstrap_ci.py`,
   `python scripts/data_checks.py`.
