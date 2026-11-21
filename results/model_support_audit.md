# Finnish ASR baseline coverage

Checked on 2026-11-21. Published model cards describe language support; the project's saved predictions establish which experiments actually completed. A model family's multilingual pretraining is not evidence of a usable Finnish ASR head.

## Completed baseline coverage

| Required method or checkpoint | Completed system | Evidence |
|---|---|---|
| MFCC + GMM-HMM | Locally trained Kaldi monophone system | [Executed notebook](../scripts/eval_gmm_hmm.ipynb), [recipe](run_metadata/gmm_hmm/gmm_recipe.json) |
| CTC encoder / wav2vec 2.0 + CTC | `jonatasgrosman/wav2vec2-large-xlsr-53-finnish` | [Finnish ASR model card](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-finnish) |
| Attention encoder-decoder / Whisper | `openai/whisper-small` | [Multilingual Whisper card](https://huggingface.co/openai/whisper-small) |
| Canary | `nvidia/canary-1b-v2` | [Model card](https://huggingface.co/nvidia/canary-1b-v2) |
| Parakeet TDT | `nvidia/parakeet-tdt-0.6b-v3` | [Finnish support and TDT architecture](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Qwen3-ASR, either listed size | `Qwen/Qwen3-ASR-1.7B` | [Model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |

All six baselines have saved predictions on the three project test sets. The group's Whisper fine-tune is reported separately. The CTC and attention-family entries reuse the relevant completed systems; they do not imply additional experiments or inflate the number of systems.

## HuBERT with a Finnish ASR head

- [`utter-project/mHuBERT-147`](https://huggingface.co/utter-project/mHuBERT-147) is a multilingual feature-extraction checkpoint. Its [configuration at revision 7ad3fc0bc5106c58c9c13526abccad527150d135](https://huggingface.co/utter-project/mHuBERT-147/blob/7ad3fc0bc5106c58c9c13526abccad527150d135/config.json) declares `HubertModel`, not `HubertForCTC`. It cannot be reported as a ready-made Finnish transcription baseline.
- [`facebook/hubert-large-ls960-ft`](https://huggingface.co/facebook/hubert-large-ls960-ft) has a CTC ASR head, but its card describes English LibriSpeech fine-tuning. It is not a verified Finnish ASR checkpoint.
- The bounded model search used `site:huggingface.co "hubert" "finnish" "ctc"`, `site:huggingface.co "hubert" "Finnish" "fine-tuned"`, and the publisher cards above. No suitable Finnish HuBERT+CTC checkpoint was verified in this search. This is not proof that no such model exists, and wav2vec2 does not silently replace the separately listed HuBERT checkpoint requirement.

Status: not run; publish the specific exclusions and search limitation. If a valid Finnish HuBERT+CTC checkpoint is identified, evaluate it under the same protocol.

## RNN-T remains an unrun supported family

Parakeet TDT is a transducer, but its checkpoint must not be relabeled as plain RNN-T. Two public cards identify Finnish-capable RNN-T candidates:

| Candidate | Evidence | Status in this project |
|---|---|---|
| [`samuli/mateli`](https://huggingface.co/samuli/mateli) | Finnish-only model; RNN-T decoder and loss; approximately 59.6M parameters; 16 kHz mono input; reference PyTorch inference implementation | Not run |
| [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | Cache-aware FastConformer-RNNT; Finnish `fi-FI` is in the broad-coverage, ready-to-transcribe tier | Not run |

The existence of these candidates means that an exclusion based on "no Finnish RNN-T support" is not justified. To satisfy a strict reading of the instructor's separate RNN-T entry, one suitable RNN-T system still needs evaluation. Do not claim full baseline coverage until that is done or the instructor explicitly accepts the existing TDT experiment for that entry.

For a new run, freeze the exact checkpoint revision and inference code, retain the existing references and normalization, validate on development clips, then evaluate all three full test sets without test-based tuning. Preserve empty outputs and record hardware, precision, batch size, language settings, and timing boundaries. No scores from a publisher's different test set may substitute for this experiment.

## Training-data overlap and reference integrity

The wav2vec2 publisher lists Common Voice 6.1 train/validation and CSS10. The project uses Common Voice 27.0; version differences alone do not establish that recordings are disjoint. Mateli describes mixed human and model-generated training transcripts. Such publisher training data do not replace any project ground truth: the project evaluation must continue to use its existing trusted references. Pretraining/fine-tuning overlap with all project test sets has not been ruled out for the published baselines.

Sources: [wav2vec2 training data](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-finnish), [Mateli training data and limitations](https://huggingface.co/samuli/mateli).
