# Whisper fine-tuning: paired significance analysis

Exploratory analysis of saved predictions. The existing test scores were known before this analysis was specified. The fixed contrast is our Whisper fine-tune minus its original Whisper baseline, evaluated for WER and CER on each of the three test sets. All six hypotheses belong to one Holm correction family.

| Test set | Metric | FT | Baseline | Difference (pp) | 95% cluster CI | Raw p | Holm p | Reject at 0.05 |
|---|---|---:|---:|---:|---|---:|---:|---|
| test_cv | WER | 23.93 | 30.94 | -7.00 | [-8.85, -5.60] | 0.00001 | 0.00006 | yes |
| test_cv | CER | 4.61 | 6.71 | -2.10 | [-2.94, -1.55] | 0.00001 | 0.00006 | yes |
| test_parl | WER | 24.72 | 25.68 | -0.96 | [-2.46, 0.26] | 0.19932 | 0.19932 | no |
| test_parl | CER | 5.89 | 8.40 | -2.52 | [-3.35, -1.72] | 0.00001 | 0.00006 | yes |
| test_fleurs | WER | 28.85 | 26.37 | +2.49 | [1.50, 3.48] | 0.00001 | 0.00006 | yes |
| test_fleurs | CER | 5.97 | 5.26 | +0.71 | [0.34, 1.12] | 0.00057 | 0.00114 | yes |

## Method

- Compare corpus edit totals, never means of clip-level WER/CER. References and denominators are identical between systems.
- Swap both systems' outputs together within each independent group: 187 CV speakers, 21 Parliament speakers, and 348 FLEURS normalized reference sentences. FLEURS groups are not speakers; speaker dependence remains unmeasured.
- The null assumes the two systems' labels are exchangeable within each group, with groups independent. The two-sided statistic is the absolute corpus error-rate difference. Equality counts as extreme.
- Use 99,999 Monte Carlo sign assignments per hypothesis, seed 20261121 plus hypothesis index. The p-value is (extreme assignments + 1) / (assignments + 1), so it is never reported as zero. The script enumerates all assignments instead when the requested budget permits an exact test.
- Holm controls the family-wise error rate across these six tests, not every pair in the bootstrap table. The reproduced 95% bootstrap intervals are pointwise, unadjusted intervals from the existing analysis; they are not simultaneous Holm intervals and their stars are not adjusted p-values.
- These intervals and tests measure test-sample uncertainty conditional on the fitted models. They do not account for training-seed variability, unknown pretraining overlap, or label errors.
- All seven systems' saved predictions are checked for complete unique IDs, matching references, one-pass hypothesis normalization, and agreement with committed WER/CER and word edit counts before analysis.

## Interpretation

- test_cv WER: lower error after fine-tuning; supported after Holm correction.
- test_cv CER: lower error after fine-tuning; supported after Holm correction.
- test_parl WER: lower error after fine-tuning; not established at the family-wise 0.05 threshold.
- test_parl CER: lower error after fine-tuning; supported after Holm correction.
- test_fleurs WER: higher error after fine-tuning; supported after Holm correction.
- test_fleurs CER: higher error after fine-tuning; supported after Holm correction.

## Reproduce

```bash
python -m pip install -r requirements-analysis.txt
python scripts/paired_significance.py
```

## Statistical references

- [SciPy: paired permutation tests and randomized p-values](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)
- [Statsmodels: Holm multiple-testing adjustment](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)
