"""Reproduce descriptive error counts and selected, traceable examples."""
import json
import re

from analysis_common import ROOT, hashes, load_verified, write_csv

EXAMPLES = [
    ("canary-1b-v2 baseline", "test_cv", "cv_common_voice_fi_23997002", "Word boundary",
     "jälleennäkemisestä is split into jälleen näkemisestä. The non-space character sequence is unchanged."),
    ("gmm-hmm monophone (trained by us)", "test_cv", "cv_common_voice_fi_25256326", "Word ending",
     "perhettä becomes perheenä. This is an observable ending change; assigning a grammatical case error would require linguistic/audio review."),
    ("parakeet-tdt-0.6b-v3 baseline", "test_fleurs", "fleurs_00016", "Number and abbreviation",
     "1100 becomes 1000, while jkr is expanded. The changed numeral and the formatting expansion are distinct issues."),
    ("qwen3-asr-1.7b zero-shot", "test_fleurs", "fleurs_00002", "Name and alphanumeric token",
     "rolando mendoza becomes rolandomen dosa; m16 becomes m60. Both the name spelling/boundary and the alphanumeric token differ."),
    ("wav2vec2-xlsr-53-finnish CTC baseline", "test_cv", "cv_common_voice_fi_26941487", "Word ending",
     "käsiään loses its final n. A small character edit is still a whole-word substitution under WER."),
    ("whisper-small zero-shot", "test_cv", "cv_common_voice_fi_32146345", "Word boundary",
     "jätti hän becomes jättihän and sammion becomes sammi on. The non-space character sequence is unchanged."),
    ("whisper-small fine-tuned (ours)", "test_parl", "parl_matti_saarinen_06590", "Spelling and word boundary",
     "euroopan becomes euroopa and maatalousunioni is split into maatalous unioni."),
    ("whisper-small fine-tuned (ours)", "test_fleurs", "fleurs_00016", "Number representation and spelling",
     "1100 is rendered as tuhat sata. Digit-versus-word formatting can contribute to the score; other word and boundary errors are also present. No semantic number-equivalence normalization is applied."),
]


def main():
    manifests, data, files = load_verified()
    summaries = []
    for (system, split), item in sorted(data.items()):
        p, a = item["predictions"], item["counts"]
        total = a.sum(axis=0)
        summaries.append(dict(system=system, test_set=split, clips=len(p),
            ref_words=int(total[1]), substitutions=int(total[4]), deletions=int(total[5]), insertions=int(total[6]),
            substitution_pct=100*total[4]/total[1], deletion_pct=100*total[5]/total[1], insertion_pct=100*total[6]/total[1],
            empty_hypotheses=sum(not r["hypothesis"] for r in p),
            exact_matches=sum(r["reference"] == r["hypothesis"] for r in p),
            boundary_only_clips=sum(r["reference"] != r["hypothesis"] and
                r["reference"].replace(" ", "") == r["hypothesis"].replace(" ", "") for r in p),
            digit_sequence_mismatch_clips=sum(re.findall(r"\d+", r["reference"]) !=
                re.findall(r"\d+", r["hypothesis"]) for r in p)))
    write_csv(ROOT / "results/error_summary.csv", summaries)
    selected = []
    for system, split, cid, category, note in EXAMPLES:
        p = next(r for r in data[system, split]["predictions"] if r["id"] == cid)
        selected.append(dict(system=system, test_set=split, id=cid, category=category,
            reference=p["reference"], hypothesis=p["hypothesis"], raw_hypothesis=p["raw_hypothesis"], observation=note))
    write_csv(ROOT / "results/error_examples.csv", selected)
    lines = ["# Error analysis of saved Finnish ASR predictions", "",
        "The aggregate counts cover every saved prediction from all seven systems. The examples are "
        "purposefully selected after viewing outputs to illustrate observable error types; they are not a random "
        "sample and do not estimate the prevalence of grammatical or semantic errors. Reference text is preserved "
        "from the fixed manifests. No new reference transcripts or audio-based judgments are introduced.", "",
        "## Aggregate observations", "",
        "S/N, D/N and I/N are percentages of total reference words. Their sum is corpus WER before rounding. "
        "Exact matches use normalized text. Boundary-only means the strings differ but become identical after removing spaces.", "",
        "| System | Test | S/N % | D/N % | I/N % | Empty hypotheses | Exact matches | Boundary-only clips |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['system']} | {r['test_set']} | {r['substitution_pct']:.2f} | {r['deletion_pct']:.2f} | "
            f"{r['insertion_pct']:.2f} | {r['empty_hypotheses']} | {r['exact_matches']} | {r['boundary_only_clips']} |")
    lines += ["", "Empty predictions remain in the evaluation and contribute deletions. Boundary-only counts describe "
        "a strict subset of errors, not all compound or word-segmentation errors. `error_summary.csv` also includes "
        "digit-sequence mismatch counts. Those counts are a surface-form diagnostic, not a semantic number-error rate: "
        "spelling a number in words can trigger a mismatch even when its value is unchanged.", "",
        "## Traceable examples", ""]
    for r in selected:
        lines += [f"### {r['category']} — {r['system']}", "", f"Test: `{r['test_set']}`; recording: `{r['id']}`.", "",
                  f"- Reference: `{r['reference']}`", f"- Hypothesis: `{r['hypothesis']}`", "", r["observation"], ""]
    lines += ["## Implications and limits", "",
        "Finnish word endings and compound boundaries make it useful to inspect character-level and word-level errors "
        "together. The examples demonstrate how a small spelling or spacing difference can affect WER; they do not "
        "establish that morphology causes most errors. No automatic morphological labels are treated as verified.", "",
        "The GMM-HMM notebook reports a train-only vocabulary of 6,235 words and OOV token rates of 36.90%, 43.68%, "
        "and 58.37% on CV, Parliament, and FLEURS respectively. This, together with monophone and grapheme modeling, "
        "is a relevant limitation rather than an isolated causal explanation of the error rate.", "",
        "Whisper fine-tuning improves aggregate CV WER/CER, improves Parliament CER, and worsens FLEURS WER/CER. "
        "The [paired analysis](paired_significance.md) supports those differences after Holm correction, while "
        "Parliament WER remains inconclusive at that threshold. Selected examples do not override corpus results.", "",
        "## Reproduce", "", "```bash", "python -m pip install -r requirements-analysis.txt",
        "python scripts/analyze_saved_errors.py", "```", "",
        "The script validates complete IDs, references, normalization, and saved scores before producing this report. "
        "The example selection is explicitly listed in the script; `error_examples.csv` also preserves each raw hypothesis."]
    (ROOT / "results/error_analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (ROOT / "results/error_analysis_metadata.json").write_text(json.dumps(dict(
        selection="purposeful illustrative examples; post hoc; no prevalence inference",
        systems=len({s for s,t in data}), predictions=sum(len(v["predictions"]) for v in data.values()),
        input_sha256=hashes(files+[ROOT/"scripts/analyze_saved_errors.py",ROOT/"scripts/analysis_common.py"])),indent=2)+"\n",encoding="utf-8")
    print(f"Validated {len(data)} system/test combinations; wrote {len(selected)} examples and {len(summaries)} aggregate rows.")


if __name__ == "__main__":
    main()
