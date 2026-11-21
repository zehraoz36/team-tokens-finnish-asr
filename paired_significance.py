"""Exploratory paired cluster randomization: Whisper FT versus its baseline.

Run from any directory: python scripts/paired_significance.py
No model execution. The six WER/CER x test-set hypotheses form ONE Holm family.
"""
import argparse
import importlib.metadata
import json
import platform
import subprocess

import numpy as np

from analysis_common import ROOT, hashes, load_verified, read_csv, write_csv

A = "whisper-small fine-tuned (ours)"
B = "whisper-small zero-shot"
SPLITS = ("test_cv", "test_parl", "test_fleurs")


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def paired_randomization(differences, permutations, seed):
    """Flip each cluster's model labels together; two-sided absolute statistic.

    Denominators are fixed and identical between models, so the sum of integer
    edit-count differences has the same tail ordering as corpus WER/CER difference.
    Enumerate exactly if 2**n <= permutations, otherwise sample with replacement
    and apply the conservative (extreme + 1)/(permutations + 1) correction.
    """
    d = np.asarray(differences, dtype=np.int64)
    if d.ndim != 1 or len(d) == 0 or permutations < 1:
        raise ValueError("Nonempty cluster differences and positive permutations required")
    observed = abs(int(d.sum()))
    exact = len(d) < 63 and 2 ** len(d) <= permutations
    total = 2 ** len(d) if exact else permutations
    extreme = 0
    rng = np.random.default_rng(seed)
    for start in range(0, total, 2048):
        size = min(2048, total - start)
        if exact:
            numbers = np.arange(start, start + size, dtype=np.uint64)
            bits = ((numbers[:, None] >> np.arange(len(d), dtype=np.uint64)) & 1).astype(np.int64)
        else:
            bits = rng.integers(0, 2, size=(size, len(d)), dtype=np.int64)
        null = (2 * bits - 1) @ d
        extreme += int(np.count_nonzero(np.abs(null) >= observed))
    p = extreme / total if exact else (extreme + 1) / (total + 1)
    return p, extreme, total, "exact" if exact else "monte_carlo"


def aggregate_clusters(manifest, difference, split):
    keys = [r["text"] if split == "test_fleurs" else r["speaker"] for r in manifest]
    if any(not k for k in keys):
        raise ValueError("Missing cluster identifier")
    groups = {}
    for k, d in zip(keys, difference):
        groups[k] = groups.get(k, 0) + int(d)
    return np.array(list(groups.values()), dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=99999)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    manifests, data, files = load_verified()
    ci_path = ROOT / "results/bootstrap_ci.csv"
    ci = {(r["test_set"], r["comparison"]): r for r in read_csv(ci_path) if r["kind"] == "difference"}
    rows = []
    for split in SPLITS:
        a, b = data[A, split]["counts"], data[B, split]["counts"]
        for metric, err, den in (("wer", 0, 1), ("cer", 2, 3)):
            if not np.array_equal(a[:, den], b[:, den]):
                raise ValueError("Reference denominators differ")
            d = aggregate_clusters(manifests[split], a[:, err] - b[:, err], split)
            seed = args.seed + len(rows)
            p, extreme, samples, method = paired_randomization(d, args.permutations, seed)
            delta = 100 * (a[:, err].sum() - b[:, err].sum()) / a[:, den].sum()
            prior = ci[split, f"{A} − {B}"]
            if round(delta, 2) != float(prior[metric]):
                raise ValueError("Bootstrap table is stale")
            rows.append(dict(test_set=split, metric=metric, model_a=A, model_b=B,
                a_percent=100 * a[:, err].sum() / a[:, den].sum(),
                b_percent=100 * b[:, err].sum() / b[:, den].sum(),
                difference_pp=delta, ci95_cluster=prior[f"{metric}_ci_cluster"].replace(" *", ""),
                cluster_unit="normalized_sentence" if split == "test_fleurs" else "speaker",
                n_clips=len(a), n_clusters=len(d), p_raw=p, extreme_count=extreme,
                permutations=samples, method=method, seed=seed))
    for r, adjusted in zip(rows, holm([r["p_raw"] for r in rows])):
        r.update(p_holm=float(adjusted), reject_holm_005=bool(adjusted <= .05), holm_family_size=6)
    write_csv(ROOT / "results/paired_significance.csv", rows)
    lines = ["# Whisper fine-tuning: paired significance analysis", "",
        "Exploratory analysis of saved predictions. The existing test scores were known before this analysis was specified. "
        "The fixed contrast is our Whisper fine-tune minus its original Whisper baseline, evaluated for WER and CER "
        "on each of the three test sets. All six hypotheses belong to one Holm correction family.", "",
        "| Test set | Metric | FT | Baseline | Difference (pp) | 95% cluster CI | Raw p | Holm p | Reject at 0.05 |",
        "|---|---|---:|---:|---:|---|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['test_set']} | {r['metric'].upper()} | {r['a_percent']:.2f} | {r['b_percent']:.2f} | "
            f"{r['difference_pp']:+.2f} | {r['ci95_cluster']} | {r['p_raw']:.5f} | {r['p_holm']:.5f} | "
            f"{'yes' if r['reject_holm_005'] else 'no'} |")
    lines += ["", "## Method", "",
        "- Compare corpus edit totals, never means of clip-level WER/CER. References and denominators are identical between systems.",
        "- Swap both systems' outputs together within each independent group: 187 CV speakers, 21 Parliament speakers, "
        "and 348 FLEURS normalized reference sentences. FLEURS groups are not speakers; speaker dependence remains unmeasured.",
        "- The null assumes the two systems' labels are exchangeable within each group, with groups independent. "
        "The two-sided statistic is the absolute corpus error-rate difference. Equality counts as extreme.",
        f"- Use {args.permutations:,} Monte Carlo sign assignments per hypothesis, seed {args.seed} plus hypothesis index. "
        "The p-value is (extreme assignments + 1) / (assignments + 1), so it is never reported as zero. "
        "The script enumerates all assignments instead when the requested budget permits an exact test.",
        "- Holm controls the family-wise error rate across these six tests, not every pair in the bootstrap table. "
        "The reproduced 95% bootstrap intervals are pointwise, unadjusted intervals from the existing analysis; "
        "they are not simultaneous Holm intervals and their stars are not adjusted p-values.",
        "- These intervals and tests measure test-sample uncertainty conditional on the fitted models. "
        "They do not account for training-seed variability, unknown pretraining overlap, or label errors.",
        "- All seven systems' saved predictions are checked for complete unique IDs, matching references, "
        "one-pass hypothesis normalization, and agreement with committed WER/CER and word edit counts before analysis.",
        "", "## Interpretation", ""]
    for r in rows:
        result = ("lower" if r["difference_pp"] < 0 else "higher") + " error after fine-tuning"
        qualifier = "supported after Holm correction" if r["reject_holm_005"] else "not established at the family-wise 0.05 threshold"
        lines.append(f"- {r['test_set']} {r['metric'].upper()}: {result}; {qualifier}.")
    lines += ["", "## Reproduce", "", "```bash", "python -m pip install -r requirements-analysis.txt",
              "python scripts/paired_significance.py", "```", "", "## Statistical references", "",
              "- [SciPy: paired permutation tests and randomized p-values](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)",
              "- [Statsmodels: Holm multiple-testing adjustment](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)"]
    (ROOT / "results/paired_significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    scripts = [ROOT / f"scripts/{p}" for p in ("paired_significance.py", "analysis_common.py", "asr_eval.py", "bootstrap_ci.py")]
    try:
        base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        base_commit = None  # Source archives may have no .git directory; input hashes still identify the data.
    meta = dict(base_commit=base_commit,
        python=platform.python_version(), numpy=np.__version__, jiwer=importlib.metadata.version("jiwer"),
        seed=args.seed, permutations=args.permutations, family_size=6, exploratory=True,
        validated_system_test_pairs=len(data), input_sha256=hashes(files + [ci_path] + scripts))
    (ROOT / "results/paired_significance_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines[:13]))


if __name__ == "__main__":
    main()
