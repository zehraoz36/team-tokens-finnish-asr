# Team Tokens — Finnish ASR

COE025 Natural Language Processing · Project 3.
Automatic speech recognition (ASR) for Finnish: zero-shot baselines vs. a fine-tuned model,
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
**cross-domain ASR robustness evaluation** — how a model trained on one domain generalizes to
others — rather than a single-domain adaptation. The small training set (4.65 h) is a known
limitation and is part of what this cross-domain setup tests.

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
(each row has `text` = cleaned/normalized and `raw_text` = original transcript).

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
This repository holds the code, clip lists, results and predictions; the audio and the trained
model live on Hugging Face and load with one line.

- **Dataset (prepared audio + transcripts):**
  https://huggingface.co/datasets/zehraoz36/team-tokens-finnish-asr-data
  The ~3.5 GB cleaned 16 kHz audio with trusted references, used by every experiment.
- **Fine-tuned model:**
  https://huggingface.co/tia88na/whisper-small-fi-tokens
  Our fine-tuned `openai/whisper-small` (~1 GB). Load it and transcribe Finnish without retraining:

  ```python
  from transformers import WhisperForConditionalGeneration, WhisperProcessor
  model = WhisperForConditionalGeneration.from_pretrained("tia88na/whisper-small-fi-tokens")
  processor = WhisperProcessor.from_pretrained("tia88na/whisper-small-fi-tokens")
  ```

## Evaluation

WER and CER (`jiwer`). Every system is scored on the **same** test clips with the **same**
text normalization (`normalize()` in `scripts/prepare_data.ipynb`, also applied to every model's
predictions before scoring). The three test sets are reported **separately**.

All systems use one shared evaluation module, `scripts/asr_eval.py`: references are the dataset's
`text` column, `normalize()` is applied once to each raw model output, WER/CER are corpus-level
(total edits / total reference words or characters), and RTF = compute time / audio time. It writes
`results/scores/<system>.csv` and per-clip predictions `results/predictions/<system>__<test_set>.csv`
(raw and normalized hypothesis).

We report both WER and CER because Finnish is agglutinative: a single wrong suffix turns a whole
word into an error and inflates WER, while CER stays low. Metric choice follows the Hugging Face
Audio Course, Chapter 5, "Evaluation":
https://huggingface.co/learn/audio-course/en/chapter5/evaluation

## Baselines

Every pretrained model is run in standard form (no extra training by us) and is a **baseline**.
Our own fine-tuning is the **contribution**. Each model is reported with its exact checkpoint name.

| Baseline | Checkpoint | Finnish support | Status |
|---|---|---|---|
| Whisper small | `openai/whisper-small` | yes (multilingual) | done — see Results |
| Qwen3-ASR | `Qwen/Qwen3-ASR-0.6B` / `1.7B` | yes | in progress |
| wav2vec 2.0 + CTC | Finnish fine-tuned checkpoint | yes | in progress |
| Canary | `nvidia/canary-1b-v2` | check model card | in progress |
| Parakeet | `nvidia/parakeet-tdt-0.6b-v3` | check model card | in progress |

If a checkpoint does not support Finnish, it is noted here with a link to its model card and not run.

## Results — Whisper small (baseline vs fine-tuned)

Baseline = `openai/whisper-small` zero-shot. Fine-tuned = the same checkpoint trained by us on the
Finnish train split (effective batch 16, lr 1e-5, 800 steps, fp16; best checkpoint chosen on dev by WER).
Same test clips, same `normalize()`, same scoring.

| Test set | Baseline WER | Baseline CER | Fine-tuned WER | Fine-tuned CER |
|---|---:|---:|---:|---:|
| test_cv | 30.94 | 6.71 | 23.93 | 4.61 |
| test_parl | 25.68 | 8.40 | 24.72 | 5.89 |
| test_fleurs | 26.41 | 5.26 | 28.87 | 5.97 |

These are the first-run numbers, kept unchanged in `results/legacy/results.csv` (predictions: `results/legacy/predictions/`).
All systems are now scored with the shared evaluation code; the current numbers (Whisper re-scored, Qwen3-ASR 1.7B)
are in `results/scores/` and per-clip predictions in `results/predictions/`. With the shared code, FLEURS changes slightly
(baseline 26.41 → 26.37, fine-tuned 28.87 → 28.85) because the first run normalized some references twice.
Generated summaries: `results/results_table.md` (all systems, `scripts/make_results_table.py`),
`results/bootstrap_ci.md` (95% confidence intervals, `scripts/bootstrap_ci.py`) and
`results/data_checks.md` (speakers, hours, test sentences found in train, `scripts/data_checks.py`).
Training and evaluation: `scripts/finnish_asr_whisper_finetune.ipynb`.
Fine-tuned model: https://huggingface.co/tia88na/whisper-small-fi-tokens

### Discussion

These results are read as a cross-domain robustness check: the model trained on Common Voice is
strongest in-domain and degrades on the out-of-domain sets.

- **test_cv (in-domain):** largest gain (WER −7.0, CER −2.1); the training data is Common Voice.
- **test_parl (out-of-domain, spontaneous):** small WER change (−1.0) but clear CER drop (−2.5);
  harder, different domain.
- **test_fleurs (out-of-domain, read):** fine-tuning regresses (WER +2.5). Specializing on the
  training domain reduces generalization — which is why we never average across test sets.
- **WER high, CER low** throughout, as expected for an agglutinative language.

## Repository structure

- `data/` — clip lists (CSV), one per split, plus `summary.csv` (`text` = cleaned, `raw_text` = original)
- `scripts/` — `prepare_data.ipynb` (data), `finnish_asr_whisper_finetune.ipynb` (fine-tuning), `asr_eval.py` (shared evaluation), `eval_*.ipynb` (one notebook per evaluated model family)
- `results/` — `scores/` (one WER/CER/RTF file per system), `predictions/` (per-clip predictions), `legacy/` (first-run results)
- `contributions/` — per-member contribution notes

## Reproduce

1. `load_dataset("zehraoz36/team-tokens-finnish-asr-data")`
2. Open `scripts/finnish_asr_whisper_finetune.ipynb` in Colab (T4 GPU), run top to bottom.
3. Baseline, fine-tuning and the comparison table are produced in order (first-run output: `results/legacy/results.csv`).
4. Every system is then scored with `scripts/asr_eval.py`: `scripts/eval_whisper.ipynb` (Whisper baseline + fine-tuned), `scripts/eval_qwen3_asr.ipynb` (Qwen3-ASR).
