# Tuana Harmankaya

- Student ID: 2309111029
- GitHub: [@tia88na](https://github.com/tia88na)
- Repository: [zehraoz36/team-tokens-finnish-asr](https://github.com/zehraoz36/team-tokens-finnish-asr)
- Dataset: [zehraoz36/team-tokens-finnish-asr-data](https://huggingface.co/datasets/zehraoz36/team-tokens-finnish-asr-data)
- Fine-tuned model: [tia88na/whisper-small-fi-tokens](https://huggingface.co/tia88na/whisper-small-fi-tokens)

## Summary

I was responsible for the modelling side of the project. I built the shared evaluation
pipeline, ran the Whisper baseline, and trained our fine-tuned model. Concretely, I evaluated
`openai/whisper-small` zero-shot as a baseline, fine-tuned the same checkpoint on our Finnish
training split, and evaluated both on the three test sets under one identical scoring policy.
Fine-tuning reduced WER by **7.0 points on Common Voice** and lowered CER on every in-domain
test set; it slightly regressed on the out-of-domain FLEURS set, which I report and analyse
rather than hide. The fine-tuned model is published on Hugging Face so that the result is fully
reproducible without retraining.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Fine-tuning and evaluation pipeline (Google Colab, T4 GPU) | `scripts/finnish_asr_whisper_finetune.ipynb` |
| 2 | Shared evaluation harness: one `normalize()` and one `evaluate()` (WER/CER via `jiwer`) applied to every system | notebook, cells 2 and 5 |
| 3 | Baseline: `openai/whisper-small` zero-shot on all three test sets | `results/results.csv` |
| 4 | Contribution: fine-tuned `openai/whisper-small` on the training split | notebook, cell 9 |
| 5 | Baseline-vs-fine-tuned comparison on the same test clips | `results/results.csv` |
| 6 | Per-clip predictions paired with references | `results/predictions/*.csv` |
| 7 | Speaker-leak verification before training (0 shared speakers) | notebook, speaker-leak check |
| 8 | Published the fine-tuned model on Hugging Face | [tia88na/whisper-small-fi-tokens](https://huggingface.co/tia88na/whisper-small-fi-tokens) |
| 9 | Results and discussion in the project README | `README.md` |

## 1. Baseline

Checkpoint: **`openai/whisper-small`** (multilingual, not `.en`), zero-shot, with no additional
training by our group. An advanced pretrained model still counts as a baseline, so it is reported
as such and kept separate from my contribution. Scored on the three test sets with the shared
`normalize()` and `jiwer`.

## 2. Contribution — fine-tuning (my own training)

I fine-tuned `openai/whisper-small` on the Finnish training split:

| Setting | Value |
|---|---|
| Effective batch size | 16 (batch 8 × gradient accumulation 2) |
| Learning rate | 1e-5, 100 warmup steps |
| Training steps | 800 |
| Precision | fp16, gradient checkpointing (fits a T4) |
| Checkpoint selection | lowest WER on the dev split (`load_best_model_at_end`) |
| Training time | ≈ 48 min |
| Final training loss | ≈ 0.51 |
| Dev WER / CER | ≈ 29.3 / 7.5 |

Predictions are normalized with the **same** `normalize()` as every other system before scoring.

## 3. Results (WER / CER, %)

| Test set | Baseline WER | Baseline CER | Fine-tuned WER | Fine-tuned CER |
|---|---:|---:|---:|---:|
| test_cv | 30.94 | 6.71 | **23.93** | **4.61** |
| test_parl | 25.68 | 8.40 | **24.72** | **5.89** |
| test_fleurs | 26.41 | 5.26 | 28.87 | 5.97 |

The three test sets are reported separately, as required. Full table: `results/results.csv`;
per-clip predictions: `results/predictions/`.

## 4. Discussion

- **test_cv (in-domain):** largest gain (WER −7.0, CER −2.1). The training data comes from
  Common Voice, so the model improves most on the Common Voice test set.
- **test_parl:** WER barely moves (−1.0) but CER drops clearly (−2.5). Parliament speech is
  harder and a different domain; the model spells better but still makes word-level errors.
- **test_fleurs (out-of-domain):** fine-tuning **hurts** here (WER +2.5). FLEURS is not in the
  training data, so specializing on Common Voice + Parliament reduces generalization. This is why
  we do not average across test sets.
- **WER is high while CER is low** across the board: Finnish is agglutinative, so a single wrong
  suffix makes a whole word wrong and inflates WER, while CER stays low. This motivates reporting
  both metrics (Hugging Face Audio Course, Chapter 5, "Evaluation").

## 5. Limitations

- Fine-tuning specializes to the training domains (Common Voice + Parliament) and regresses on FLEURS.
- The training set is small (4.65 h, few speakers), which caps how much fine-tuning can help.
- Training was stopped at 800 steps for compute and time; longer training with early stopping
  could change the picture.
- Common Voice and Parliament may appear in the pre-training data of the checkpoint; FLEURS is
  the cleaner comparison.

## 6. Reproducibility

Anyone can reproduce the pipeline by opening `scripts/finnish_asr_whisper_finetune.ipynb` in
Colab (T4 GPU) and running it top to bottom; the dataset loads from Hugging Face automatically.
The trained model can be used directly without retraining:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
model = WhisperForConditionalGeneration.from_pretrained("tia88na/whisper-small-fi-tokens")
processor = WhisperProcessor.from_pretrained("tia88na/whisper-small-fi-tokens")
```

## Deliverables

- `scripts/finnish_asr_whisper_finetune.ipynb` — baseline, fine-tuning and evaluation
- `results/results.csv` — baseline vs fine-tuned WER/CER
- `results/predictions/*.csv` — per-clip reference vs hypothesis
- README "Results" and "Discussion" sections
- [tia88na/whisper-small-fi-tokens](https://huggingface.co/tia88na/whisper-small-fi-tokens) — fine-tuned model
