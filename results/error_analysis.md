# Error analysis of saved Finnish ASR predictions

The aggregate counts cover every saved prediction from all eight systems. The examples are purposefully selected after viewing outputs to illustrate observable error types; they are not a random sample and do not estimate the prevalence of grammatical or semantic errors. Reference text is preserved from the fixed manifests. No new reference transcripts or audio-based judgments are introduced.

## Aggregate observations

S/N, D/N and I/N are percentages of total reference words. Their sum is corpus WER before rounding. Exact matches use normalized text. Boundary-only means the strings differ but become identical after removing spaces.

| System | Test | S/N % | D/N % | I/N % | Empty hypotheses | Exact matches | Boundary-only clips |
|---|---|---:|---:|---:|---:|---:|---:|
| Mateli RNN-T baseline | test_cv | 11.55 | 1.97 | 1.50 | 5 | 989 | 89 |
| Mateli RNN-T baseline | test_fleurs | 12.87 | 2.16 | 2.19 | 0 | 217 | 48 |
| Mateli RNN-T baseline | test_parl | 8.35 | 2.92 | 1.34 | 1 | 476 | 55 |
| canary-1b-v2 baseline | test_cv | 8.82 | 0.66 | 0.89 | 0 | 1148 | 44 |
| canary-1b-v2 baseline | test_fleurs | 7.08 | 0.74 | 1.06 | 0 | 351 | 44 |
| canary-1b-v2 baseline | test_parl | 5.89 | 3.28 | 0.98 | 1 | 653 | 59 |
| gmm-hmm monophone (trained by us) | test_cv | 58.20 | 20.62 | 1.86 | 0 | 43 | 2 |
| gmm-hmm monophone (trained by us) | test_fleurs | 71.89 | 19.47 | 1.75 | 0 | 0 | 0 |
| gmm-hmm monophone (trained by us) | test_parl | 60.99 | 24.76 | 1.44 | 0 | 53 | 0 |
| parakeet-tdt-0.6b-v3 baseline | test_cv | 6.25 | 0.46 | 0.47 | 0 | 1316 | 31 |
| parakeet-tdt-0.6b-v3 baseline | test_fleurs | 10.81 | 1.50 | 1.04 | 0 | 226 | 42 |
| parakeet-tdt-0.6b-v3 baseline | test_parl | 8.54 | 3.58 | 1.16 | 5 | 518 | 55 |
| qwen3-asr-1.7b zero-shot | test_cv | 25.98 | 2.84 | 2.31 | 0 | 414 | 45 |
| qwen3-asr-1.7b zero-shot | test_fleurs | 20.67 | 2.55 | 1.95 | 0 | 53 | 10 |
| qwen3-asr-1.7b zero-shot | test_parl | 20.29 | 3.63 | 2.57 | 0 | 180 | 26 |
| wav2vec2-xlsr-53-finnish CTC baseline | test_cv | 34.99 | 3.66 | 4.14 | 0 | 353 | 52 |
| wav2vec2-xlsr-53-finnish CTC baseline | test_fleurs | 47.51 | 4.64 | 7.37 | 0 | 3 | 2 |
| wav2vec2-xlsr-53-finnish CTC baseline | test_parl | 43.77 | 6.06 | 6.39 | 0 | 37 | 9 |
| whisper-small fine-tuned (ours) | test_cv | 19.62 | 2.07 | 2.24 | 0 | 609 | 65 |
| whisper-small fine-tuned (ours) | test_fleurs | 23.06 | 2.50 | 3.30 | 0 | 47 | 7 |
| whisper-small fine-tuned (ours) | test_parl | 18.23 | 3.68 | 2.82 | 0 | 189 | 39 |
| whisper-small zero-shot | test_cv | 24.77 | 2.44 | 3.72 | 0 | 444 | 68 |
| whisper-small zero-shot | test_fleurs | 20.97 | 2.32 | 3.08 | 0 | 46 | 23 |
| whisper-small zero-shot | test_parl | 15.97 | 6.06 | 3.65 | 0 | 180 | 43 |

Empty predictions remain in the evaluation and contribute deletions. Boundary-only counts describe a strict subset of errors, not all compound or word-segmentation errors. `error_summary.csv` also includes digit-sequence mismatch counts. Those counts are a surface-form diagnostic, not a semantic number-error rate: spelling a number in words can trigger a mismatch even when its value is unchanged.

## Traceable examples

### Word boundary — canary-1b-v2 baseline

Test: `test_cv`; recording: `cv_common_voice_fi_23997002`.

- Reference: `ne olivat aina yhtä iloisia jälleennäkemisestä`
- Hypothesis: `ne olivat aina yhtä iloisia jälleen näkemisestä`

jälleennäkemisestä is split into jälleen näkemisestä. The non-space character sequence is unchanged.

### Word ending — gmm-hmm monophone (trained by us)

Test: `test_cv`; recording: `cv_common_voice_fi_25256326`.

- Reference: `perhettä hänellä ei ollut koskaan ollutkaan`
- Hypothesis: `perheenä hänellä ei ollut koskaan ollutkaan`

perhettä becomes perheenä. This is an observable ending change; assigning a grammatical case error would require linguistic/audio review.

### Number and abbreviation — parakeet-tdt-0.6b-v3 baseline

Test: `test_fleurs`; recording: `fleurs_00016`.

- Reference: `sen maine ylellisyyden keskipisteenä alkoi noin vuonna 400 jkr ja jatkui noin vuoteen 1100 asti`
- Hypothesis: `sen maine ylellisyyden keskipisteenä alkoi noin vuonna 400 jälkeen kristuksen ja jatkui noin vuoteen 1000 asti`

1100 becomes 1000, while jkr is expanded. The changed numeral and the formatting expansion are distinct issues.

### Name and alphanumeric token — qwen3-asr-1.7b zero-shot

Test: `test_fleurs`; recording: `fleurs_00002`.

- Reference: `rolando mendoza ampui matkailijoita päin m16 kiväärillä`
- Hypothesis: `rolandomen dosa ampui matkalieota pain m60 kivärillä`

rolando mendoza becomes rolandomen dosa; m16 becomes m60. Both the name spelling/boundary and the alphanumeric token differ.

### Word ending — wav2vec2-xlsr-53-finnish CTC baseline

Test: `test_cv`; recording: `cv_common_voice_fi_26941487`.

- Reference: `hänkin vain levitteli käsiään`
- Hypothesis: `hänkin vain levitteli käsiää`

käsiään loses its final n. A small character edit is still a whole-word substitution under WER.

### Word boundary — whisper-small zero-shot

Test: `test_cv`; recording: `cv_common_voice_fi_32146345`.

- Reference: `siihen jätti hän sammion toistaiseksi`
- Hypothesis: `siihen jättihän sammi on toistaiseksi`

jätti hän becomes jättihän and sammion becomes sammi on. The non-space character sequence is unchanged.

### Spelling and word boundary — whisper-small fine-tuned (ours)

Test: `test_parl`; recording: `parl_matti_saarinen_06590`.

- Reference: `euroopan unioni on lähinnä maatalousunioni`
- Hypothesis: `euroopa unioni on lähinnä maatalous unioni`

euroopan becomes euroopa and maatalousunioni is split into maatalous unioni.

### Number representation and spelling — whisper-small fine-tuned (ours)

Test: `test_fleurs`; recording: `fleurs_00016`.

- Reference: `sen maine ylellisyyden keskipisteenä alkoi noin vuonna 400 jkr ja jatkui noin vuoteen 1100 asti`
- Hypothesis: `sen maineyellisyyden keskipisteenä alkoi noin vuonna 400 jälkeen kristuksen ja jatkuin noin vuoteen tuhat sata aste`

1100 is rendered as tuhat sata. Digit-versus-word formatting can contribute to the score; other word and boundary errors are also present. No semantic number-equivalence normalization is applied.

### Name and spelling — Mateli RNN-T baseline

Test: `test_fleurs`; recording: `fleurs_00000`.

- Reference: `yhdysvaltojen ilmailuviranomainen väittää että nextgen on järjestelmä jonka ansiosta lentokoneet voivat lentää lyhyempiä reittejä säästää vuosittain miljoonia gallonia polttoainetta ja vähentää hiilidioksidipäästöjä`
- Hypothesis: `yhdysvaltojen ilmailuviranomainen väittää että nekst jen on järjestelmä jonka ansiosta lentokoneet voivat lentää lyhyempiä reittejä säästää vuosittain miljoonia kallonia polttoainetta ja vähentää hiilidioksidipäästöjä`

NextGen is rendered phonetically as nekst jen; gallonia becomes kallonia. These are observable spelling and token-boundary differences, not verified semantic errors.

## Implications and limits

Finnish word endings and compound boundaries make it useful to inspect character-level and word-level errors together. The examples demonstrate how a small spelling or spacing difference can affect WER; they do not establish that morphology causes most errors. No automatic morphological labels are treated as verified.

The GMM-HMM notebook reports a train-only vocabulary of 6,235 words and OOV token rates of 36.90%, 43.68%, and 58.37% on CV, Parliament, and FLEURS respectively. This, together with monophone and grapheme modeling, is a relevant limitation rather than an isolated causal explanation of the error rate.

Whisper fine-tuning improves aggregate CV WER/CER, improves Parliament CER, and worsens FLEURS WER/CER. The [paired analysis](paired_significance.md) supports those differences after Holm correction, while Parliament WER remains inconclusive at that threshold. Selected examples do not override corpus results.

## Reproduce

```bash
python -m pip install -r requirements-analysis.txt
python scripts/analyze_saved_errors.py
```

The script validates complete IDs, references, normalization, and saved scores before producing this report. The example selection is explicitly listed in the script; `error_examples.csv` also preserves each raw hypothesis.
