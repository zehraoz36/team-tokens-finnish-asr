# Begüm Ziyade

- Student ID: 220901561
- GitHub: [@bgmzyd](https://github.com/bgmzyd)
- Repository: [zehraoz36/team-tokens-finnish-asr](https://github.com/zehraoz36/team-tokens-finnish-asr)
- Dataset: [zehraoz36/team-tokens-finnish-asr-data](https://huggingface.co/datasets/zehraoz36/team-tokens-finnish-asr-data)

## Summary

I was responsible for the shared evaluation of all systems. I wrote one evaluation module,
`scripts/asr_eval.py`, that every model in the project is scored with, so that all results use
the same test clips, the same references, one text normalization policy and the same WER/CER
computation, as the project rules require. With it I re-scored the Whisper baseline and our
fine-tuned Whisper model, saving raw and normalized per-clip predictions, exact checkpoint
revisions and real-time factors, and I set up and ran the Qwen3-ASR baseline. I also checked the
scoring independently and fixed a normalization inconsistency in the earlier evaluation.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Shared evaluation module used by every system | `scripts/asr_eval.py` (`dd0994d`, `7c2175d`) |
| 2 | Found and fixed a normalization inconsistency (two different `normalize()` copies; references normalized twice) | section 2 |
| 3 | Whisper baseline and fine-tuned model re-scored with the shared code | `scripts/eval_whisper.ipynb`, `results/scores/whisper-*.csv` (`f404f8c`) |
| 4 | Raw + normalized per-clip predictions for both Whisper systems | `results/predictions/whisper-*__*.csv` |
| 5 | Qwen3-ASR baseline notebook (Finnish support check, dev-clip pipeline check) and run | `scripts/eval_qwen3_asr.ipynb` (`bde4dc3`, `10280fa`) |
| 6 | Checkpoint revision, hardware, precision and RTF recorded for every run | `results/scores/*.csv` |
| 7 | Independent check of the WER/CER numbers | section 4 |
| 8 | Earlier first-run results kept unchanged for comparison | `results/legacy/` (`dd0994d`) |

## 1. Shared evaluation module (`scripts/asr_eval.py`)

One file that every teammate imports, so no system is scored differently:

| Item | Policy |
|---|---|
| Test sets | `test_cv`, `test_parl`, `test_fleurs`, always reported separately, never averaged |
| References | the dataset's `text` column (= `normalize(raw_text)` from `scripts/prepare_data.ipynb`) |
| Hypotheses | `normalize(raw model output)`, applied exactly once |
| WER / CER | corpus-level with `jiwer`: total edits / total reference words (characters, spaces included, for CER); not an average of per-clip scores |
| Speed | RTF = compute time / audio time, timing only the transcription call (feature extraction + decoding) on audio already in memory; RTFx = 1 / RTF |
| Outputs | `results/scores/<system>.csv` (one row per test set, with checkpoint, revision, hardware, precision, batch size, settings) and `results/predictions/<system>__<test_set>.csv` (id, reference, hypothesis, raw hypothesis) |

Design choices:
- **One score file per system**, so teammates running different models at the same time never
  edit the same file (no merge conflicts).
- **Resumable runs:** a test set that is already scored is skipped, so a Colab disconnect does not
  force a full re-run.
- A new model only needs a `transcribe_fn(list of 16 kHz arrays) -> list of strings` and one call
  to `evaluate_system(...)`.

## 2. Normalization fix

The first evaluation had two copies of `normalize()` (data preparation and the fine-tuning
notebook) with different regular expressions for thousands separators and brackets, and it
normalized references that were already normalized. Normalizing twice merges numbers
(`150–200` → `150 200` → `150200`). `asr_eval.normalize()` is now the data-preparation policy
(identical output on all 21,361 references and Whisper outputs I tested), applied once. The only
effect on reported numbers is FLEURS: 26.41 → 26.37 (baseline) and 28.87 → 28.85 (fine-tuned).

## 3. Results (WER / CER, %)

Same 4,662 test clips (11.45 h) and scoring for every system. Hardware: Colab Tesla T4.

**Baselines (pretrained, no training by us)**

| System | Checkpoint (revision) | test_cv | test_parl | test_fleurs | RTF |
|---|---|---|---|---|---|
| Whisper small, zero-shot | `openai/whisper-small` (`973afd2`) | 30.94 / 6.71 | 25.68 / 8.40 | 26.37 / 5.26 | 0.011–0.018 |
| Qwen3-ASR 1.7B, zero-shot | `Qwen/Qwen3-ASR-1.7B` (`7278e1e`) | 31.13 / 7.23 | 26.50 / 6.94 | **25.16** / 5.81 | 0.037–0.051 |

**Our contribution (trained by our group)**

| System | Checkpoint (revision) | test_cv | test_parl | test_fleurs | RTF |
|---|---|---|---|---|---|
| Whisper small, fine-tuned | `tia88na/whisper-small-fi-tokens` (`58b213c`) | **23.93 / 4.61** | **24.72 / 5.89** | 28.85 / 5.97 | 0.010–0.017 |

- Qwen3-ASR 1.7B is the best system on the out-of-domain FLEURS set (25.16 WER), but it is not
  better than zero-shot Whisper small on Common Voice or Parliament, and our fine-tuned Whisper
  small beats it on both in-domain sets. It is also 2–4× slower on the T4 (RTF 0.037–0.051 vs
  0.010–0.018).

Settings: Whisper uses language = Finnish, task = transcribe, greedy decoding, max 225 new
tokens, fp16, batch 16, and the `openai/whisper-small` processor for both systems (the one used for
fine-tuning). Qwen3-ASR uses forced `language="Finnish"`, greedy decoding, max 256 new tokens,
`qwen-asr` 0.0.6 with `transformers` 4.57.6, float16, batch 16.

## 4. Checks

- **Scoring:** I recomputed WER and CER for all nine result files (Whisper and Qwen) with my own edit-distance
  code instead of `jiwer`; every value matches the score files exactly.
- **Predictions:** every prediction file has the same clip IDs and references as the dataset, in the
  same order, and no empty hypotheses.
- **Reproduction:** re-scoring the earlier saved fine-tuned predictions with the shared code gives
  the earlier numbers (23.93 / 24.72), confirming the new code changes only the normalization detail
  above.
- **Language support:** Finnish is in `SUPPORTED_LANGUAGES` of the `qwen-asr` package (checked in the
  package source and asserted in the notebook); Whisper small is a multilingual (not `.en`)
  checkpoint and is run with the language forced to Finnish.
- **Qwen precision:** Qwen ran in float16 on the T4. I checked on 400 development clips that float32
  gives the same WER (31.82 vs 31.82, 390/400 identical outputs), so the float16 result stands
  (`scripts/qwen_precision_check.ipynb`, `results/qwen_precision_check.csv`).
- **Test set use:** no setting was tuned on the test sets. The Qwen pipeline check uses 8
  development clips only.

## 5. Limitations

- RTF depends on the GPU, batch size and precision; numbers are comparable only within the same
  hardware (Colab T4).
- Only the Qwen3-ASR 1.7B checkpoint was run; the 0.6B run is available in the notebook (`RUN_06B = True`) but was not needed.
- Whisper zero-shot shows a few long hallucinated outputs on Parliament speech (one clip over 150
  words); these are counted as errors, not removed.

## Commit evidence

Repository [zehraoz36/team-tokens-finnish-asr](https://github.com/zehraoz36/team-tokens-finnish-asr), branch `main`:

- [`dd0994d`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/dd0994d) (2026-11-19) – shared evaluation code, Whisper and Qwen3-ASR evaluation notebooks, first-run results moved to `results/legacy/`
- [`ec0741c`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/ec0741c) (2026-11-19) – Whisper evaluation uses the base processor for both systems
- [`7c2175d`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/7c2175d) (2026-11-19) – reliable model download on Colab; resumable evaluation
- [`f404f8c`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/f404f8c) (2026-11-19) – Whisper baseline and fine-tuned results with the shared evaluation
- [`bde4dc3`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/bde4dc3), [`10280fa`](https://github.com/zehraoz36/team-tokens-finnish-asr/commit/10280fa) (2026-11-19) – Qwen3-ASR notebook: float16 on T4, GPU check, optional Google Drive saving
- Qwen3-ASR 1.7B results (`results/scores/qwen3-asr-1-7b-zero-shot.csv`, `results/predictions/qwen3-asr-1-7b-zero-shot__*.csv`) and this note
