# Ece Mina Örenler

- GitHub: [@eceorenler](https://github.com/eceorenler)
- Repository: [zehraoz36/team-tokens-finnish-asr](https://github.com/zehraoz36/team-tokens-finnish-asr)

## Summary

I carried out the classical GMM-HMM experiment and evaluated four additional Finnish ASR baselines: wav2vec 2.0 + CTC, Canary 1B v2, Parakeet TDT 0.6B v3, and Mateli RNN-T. I ran the experiments in Google Colab, worked through installation and execution problems, collected the completed outputs, and uploaded the evaluation notebooks, scores, predictions, and run metadata to the shared repository.

All five systems were evaluated on the project's three fixed test sets: Common Voice (1,833 recordings), Parliament (1,913 recordings), and FLEURS (916 recordings). Each system therefore has predictions for all 4,662 test recordings.

## Overview of my contributions

| Contribution | Evidence |
|---|---|
| Train and evaluate a Finnish monophone GMM-HMM baseline | [GMM-HMM notebook](../scripts/eval_gmm_hmm.ipynb) |
| Evaluate a published Finnish wav2vec 2.0 + CTC checkpoint | [wav2vec2 notebook](../scripts/eval_wav2vec2.ipynb) |
| Evaluate Canary 1B v2 with Finnish source and target language settings | [Canary notebook](../scripts/eval_canary.ipynb) |
| Evaluate the multilingual Parakeet TDT 0.6B v3 checkpoint | [Parakeet notebook](../scripts/eval_parakeet.ipynb) |
| Evaluate the published Finnish Mateli RNN-T baseline | [Mateli notebook](../scripts/eval_mateli_rnnt.ipynb) |
| Preserve scores, raw and normalized predictions, and experiment settings | [Scores](../results/scores/), [predictions](../results/predictions/), [run metadata](../results/run_metadata/) |
| Upload the five completed evaluation notebooks and their result files | Commits linked below |

## 1. GMM-HMM training and evaluation

I ran Kaldi training on the project's Common Voice training split, containing 3,770 recordings and approximately 4.65 hours of speech. The system uses 13-dimensional MFCC features with deltas, per-utterance mean normalization, grapheme-based monophone acoustic modeling, and a vocabulary and unigram language model derived only from the training transcripts.

I worked through the Kaldi installation problems encountered in Colab, including the missing LAPACK development dependency and the LAPACK call compatibility issue. After training, I ran the development smoke test and all three full test evaluations, and saved the results and a training backup.

This system is a classical baseline trained by our group. Under the project rules, it remains a baseline, while the training work is part of my contribution.

## 2. Pretrained ASR evaluations

I ran the following published checkpoints without additional training:

| System | Checkpoint | Evaluation setup |
|---|---|---|
| wav2vec 2.0 + CTC | `jonatasgrosman/wav2vec2-large-xlsr-53-finnish` | Finnish ASR head, greedy CTC decoding, no external language model |
| Canary | `nvidia/canary-1b-v2` | Finnish source and target language, checkpoint-default decoding |
| Parakeet | `nvidia/parakeet-tdt-0.6b-v3` | Multilingual TDT model, automatic language recognition, checkpoint-default decoding |
| Mateli RNN-T | `samuli/mateli` | Finnish-only published checkpoint, greedy RNN-T decoding, no external language model |

The neural evaluations ran on a Tesla T4. Wav2vec2 used float32 with batch size 4; Canary, Parakeet, and Mateli used float32 with batch size 1. Exact checkpoint revisions and effective settings are recorded in score files and run metadata. Parakeet is a Token-and-Duration Transducer (TDT), while Mateli is the separately evaluated plain RNN-T baseline.

## 3. Results

WER and CER are corpus-level percentages. Each cell shows WER / CER.

| System | test_cv | test_parl | test_fleurs |
|---|---:|---:|---:|
| GMM-HMM monophone, trained by us | 80.69 / 42.58 | 87.20 / 47.86 | 93.11 / 53.07 |
| wav2vec2 + Finnish CTC | 42.78 / 8.93 | 56.22 / 13.07 | 59.52 / 15.83 |
| Canary 1B v2 | 10.36 / 2.00 | 10.14 / 3.65 | 8.88 / 2.10 |
| Parakeet TDT 0.6B v3 | 7.17 / 1.37 | 13.28 / 4.09 | 13.35 / 2.72 |
| Mateli RNN-T baseline | 15.02 / 3.12 | 12.61 / 3.77 | 17.21 / 6.96 |

The evaluations use the team's shared [evaluation module](../scripts/asr_eval.py). References retain the manifest's normalized text, and raw hypotheses are normalized once before scoring. The three test sets remain separate. Empty hypotheses are retained and scored rather than removed.

Parakeet has the lowest WER among the five systems on Common Voice; Canary has the lowest WER on Parliament and FLEURS. Mateli's RNN-T results are reported as a baseline, not as team training. The high GMM-HMM error rates are retained as baseline findings. Its small training set, monophone acoustic model, grapheme pronunciation approximation, and closed training vocabulary are important limitations.

## 4. Reproducibility and limitations

- The uploaded notebooks preserve the executed experiment outputs and describe the experiments in English.
- Scores include checkpoint or model identifiers, revisions, hardware, precision, batch size, and timing settings. Per-recording predictions support subsequent rescoring and statistical analysis.
- GMM-HMM ran on CPU. Its reported timing excludes precomputed MFCC extraction, so its RTF is not directly comparable with the neural GPU timings.
- Mateli inference compute time totaled 3,551.4 seconds (about 59.2 minutes) across the three test sets on a T4. RTF timing covers the shared evaluator's transcription call, not model/data loading.
- The executed GMM-HMM notebook preserves the installation troubleshooting order. A clean-runtime installation sequence still needs validation; the completed results do not establish that every setup cell can be rerun in its displayed order.

## Commit evidence

- [967bca4](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/967bca4ce78b3b9c2494eedd5d3040ce7fd10d5d): evaluation notebooks for GMM-HMM, wav2vec2, Canary, and Parakeet.
- [166500e](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/166500ed7bffe2a515d057b57b05b202760b6e26): verified scores and predictions for the four baselines.
- [059b86a](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/059b86ac78c6a43b6716b152ff49919b2e143a04): configurations and run metadata for the four baselines.
