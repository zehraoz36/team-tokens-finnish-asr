# Tuana Harmankaya

- Student ID: 2309111029
- GitHub: [@tia88na](https://github.com/tia88na)
- Repository: [zehraoz36/team-tokens-finnish-asr](https://github.com/zehraoz36/team-tokens-finnish-asr)
- Dataset: [zehraoz36/team-tokens-finnish-asr-data](https://huggingface.co/datasets/zehraoz36/team-tokens-finnish-asr-data)

## Summary

I built the modelling side of the project: a shared evaluation harness, the Whisper baseline, and our own fine-tuned model. I ran `openai/whisper-small` zero-shot as a baseline, fine-tuned the same checkpoint on our Finnish training split, and evaluated both on the three test sets with one identical scoring policy. Fine-tuning lowered WER by **7 points on Common Voice** and improved CER on every in-domain set, while slightly regressing on the out-of-domain FLEURS set — a result we report honestly and discuss.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Fine-tuning + evaluation pipeline (Google Colab, T4) | `scripts/finetune_whisper.ipynb` |
| 2 | Shared evaluation harness: one `normalize()` + `evaluate()` (WER/CER via jiwer) for every system | `scripts/finetune_whisper.ipynb`, cells 2 and 5 |
| 3 | Baseline: `openai/whisper-small` zero-shot on all three test sets | `results/results.csv` |
| 4 | Contribution: fine-tuned `openai/whisper-small` on the train split | `scripts/finetune_whisper.ipynb`, cell 9 |
| 5 | Baseline-vs-fine-tuned results on the same test clips | `results/results.csv` |
| 6 | Speaker-leak verification before training (0 shared speakers) | notebook, speaker-leak check cell |

## 1. Baseline

Checkpoint: **`openai/whisper-small`** (multilingual, not `.en`), zero-shot, no extra training.
Scored on the three test sets with the shared `normalize()` and jiwer (WER/CER). An advanced
pretrained model counts as a baseline, so this is reported as a baseline, not a contribution.

## 2. Contribution — fine-tuning (my own training)

Fine-tuned `openai/whisper-small` on the Finnish training split:

- Effective batch 16 (batch 8 × grad-accum 2), learning rate 1e-5, 100 warmup steps, 800 steps.
- `fp16`, gradient checkpointing (fits a free/standard T4).
- Checkpoint selected on the dev split by lowest WER (`load_best_model_at_end`).
- Predictions normalized with the **same** `normalize()` as every other system before scoring.
- Training time ≈ 48 min; final training loss ≈ 0.51; dev WER ≈ 29.3, dev CER ≈ 7.5.

The trained model is saved separately (not on GitHub because of its size).

## 3. Results (WER / CER, %)

| Test set | Baseline WER | Baseline CER | Fine-tuned WER | Fine-tuned CER |
|---|---:|---:|---:|---:|
| test_cv | 30.94 | 6.71 | **23.93** | **4.61** |
| test_parl | 25.68 | 8.40 | **24.72** | **5.89** |
| test_fleurs | 26.41 |
