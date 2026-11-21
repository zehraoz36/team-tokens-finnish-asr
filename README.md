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

### Split checks

The committed speaker identifiers show **0 overlap between train and the CV/Parliament
development and test sets**. This checks the available identifiers rather than proving identity
disjointness across unrelated corpora. FLEURS provides no speaker IDs, so speaker overlap cannot
be checked for that set. Its bootstrap analysis groups recordings by normalized reference
sentence, not by speaker. Preparation: `scripts/prepare_data.ipynb`; checks: `results/data_checks.md`.

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

We report both WER and CER because word-level correctness and character-level differences offer
complementary views. A small ending or spelling change can count as a whole-word substitution;
CER measures the size of character edits and includes spaces. Actual examples are in
[the error analysis](results/error_analysis.md). Metric choice follows the Hugging Face
Audio Course, Chapter 5, "Evaluation":
https://huggingface.co/learn/audio-course/en/chapter5/evaluation

95% bootstrap confidence intervals (cluster and clip resampling, with paired differences) are in
`results/bootstrap_ci.md` (`scripts/bootstrap_ci.py`).

The [paired significance analysis](results/paired_significance.md) adds two-sided cluster
randomization tests for Whisper fine-tuned versus baseline. Holm correction covers the six
WER/CER-by-test-set hypotheses. This is an exploratory analysis performed after observing test
scores. The bootstrap intervals remain pointwise, unadjusted intervals.

RTF and RTFx describe recorded computation cost. GMM-HMM uses CPU and excludes precomputed
MFCC extraction; neural runs use a T4 with different precision and batching. The timing values
are not an equal-conditions speed ranking. Settings and timing boundaries are recorded per system.

## Baselines

Every pretrained model is run in standard form (no extra training by our group) and is a
**baseline**; our own fine-tuning is the **contribution**. Our trained GMM-HMM also remains a
baseline, with credit for the group's training work. Exact checkpoints and revisions are in
`results/scores/`. The [coverage audit](results/model_support_audit.md) documents the completed
systems and the remaining HuBERT checkpoint limitation.

| Method family | System (checkpoint) | Finnish support | Status |
|---|---|---|---|
| Classical MFCC + GMM-HMM | `local/kaldi-grapheme-mono` (trained by us) | trained on our data | done |
| Self-supervised + CTC | `jonatasgrosman/wav2vec2-large-xlsr-53-finnish` | yes | done |
| Self-supervised + CTC (HuBERT) | — | suitable Finnish ASR checkpoint not verified in our search | not run (see note) |
| Attention encoder-decoder | `openai/whisper-small` | yes (multilingual) | done |
| Attention encoder-decoder | `nvidia/canary-1b-v2` | yes | done |
| Transducer (TDT) | `nvidia/parakeet-tdt-0.6b-v3` | yes | done |
| Transducer (RNN-T) | `samuli/mateli` | Finnish RNN-T | done (baseline) |
| Large audio-language model | `Qwen/Qwen3-ASR-1.7B` | yes | done |

**HuBERT + CTC:** we did not verify a suitable Finnish checkpoint in the documented search.
[`utter-project/mHuBERT-147`](https://huggingface.co/utter-project/mHuBERT-147) is a feature
encoder, while [`facebook/hubert-large-ls960-ft`](https://huggingface.co/facebook/hubert-large-ls960-ft)
is an English ASR checkpoint. Neither establishes a Finnish HuBERT baseline. This is a bounded
search result, not proof that no suitable checkpoint exists.

**RNN-T:** Parakeet is TDT, not plain RNN-T. We evaluated the Finnish `samuli/mateli`
checkpoint (revision `3dd0f094234d565dc923391eee15c4425feafdea`) as a separate published
baseline on all three test sets. Results and the inference setup are in
[`results/scores/mateli-rnn-t-baseline.csv`](results/scores/mateli-rnn-t-baseline.csv),
the [Mateli notebook](scripts/eval_mateli_rnnt.ipynb), and the
[coverage audit](results/model_support_audit.md). It is not a team fine-tune.

## Results

WER / CER in %, corpus-level, same test clips, references and `normalize()` for every system.
Full table with checkpoints, revisions, RTF and hardware: `results/results_table.md`
(`scripts/make_results_table.py`). Confidence intervals: `results/bootstrap_ci.md`.

### Baselines (including our trained GMM-HMM)

| System | test_cv WER / CER | test_parl WER / CER | test_fleurs WER / CER |
|---|---:|---:|---:|
| parakeet-tdt-0.6b-v3 | **7.17** / 1.37 | 13.28 / 4.09 | 13.35 / 2.72 |
| canary-1b-v2 | 10.36 / 2.00 | **10.14** / 3.65 | **8.88** / 2.10 |
| mateli RNN-T baseline | 15.02 / 3.12 | 12.61 / 3.77 | 17.21 / 6.96 |
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

- **Strongest baselines:** Parakeet has the lowest WER on CV; Canary has the lowest WER on
  Parliament and FLEURS. Mateli is a separate Finnish RNN-T baseline; its three domain results
  are reported independently and are not averaged.
- **Our contribution:** fine-tuning `whisper-small` improves clearly over its own zero-shot baseline
  (test_cv WER 30.94 → 23.93, −7.0; test_parl CER 8.40 → 5.89) and beats the Qwen3-ASR baseline on
  the in-domain set. It does **not** reach Parakeet/Canary in these experiments. This comparison
  does not isolate the effects of model size, training data, or architecture.
- **Cross-domain behaviour:** fine-tuning helps most in-domain (test_cv), is mixed on Parliament
  (WER barely changes, CER drops clearly) and regresses on the out-of-domain FLEURS read-speech set
  (WER +2.5). The observed regression is consistent with domain specialization but does not
  establish its cause. Each test set remains separate.
- **Statistical uncertainty:** after Holm correction, CV WER/CER and Parliament CER improve,
  while FLEURS WER/CER worsen. Parliament WER is inconclusive (adjusted p = 0.19932).
  See [all six comparisons](results/paired_significance.md).
- **Error patterns:** [actual reference/hypothesis examples](results/error_analysis.md) show
  word-ending, boundary, name, and number-form differences. These illustrations are not a
  prevalence estimate or a verified morphological annotation of the corpus.
- **Limitations:** training uses only 4.65 hours and 10 speakers. Published-model pretraining
  overlap with the test corpora has not been ruled out. GMM-HMM additionally uses a closed
  train-only vocabulary, a unigram language model, and grapheme monophones.

## Repository structure

Root files: `.gitignore` documents ignored local/build artifacts; `LICENSE` covers our code;
`requirements.txt` and `requirements-analysis.txt` list runtime dependencies;
`test_paired_significance.py` contains paired-statistics unit tests. The root-level
`error_summary.csv`, `error_examples.csv`, `error_analysis_metadata.json`,
`paired_significance.csv`, and `paired_significance_metadata.json` are generated compatibility
copies of the canonical outputs under `results/`. Both analysis scripts regenerate both locations;
use the `results/` files in project reports.

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
5. Saved-prediction analysis (no GPU): install `requirements-analysis.txt`, then run
   `python scripts/paired_significance.py` and `python scripts/analyze_saved_errors.py`.
6. The original GMM notebook preserves the executed troubleshooting session. A separate
   `scripts/eval_gmm_hmm_clean.ipynb` integrates the installation fixes in execution order;
   its clean-runtime Colab build has not yet been executed. The original results remain unchanged.
