"""95% bootstrap confidence intervals for WER/CER and for differences between systems.

Uses the saved per-clip predictions (results/predictions/<system>__<test_set>.csv), so no model is
run again. For every resample, WER is recomputed corpus-level (total edits / total reference words),
exactly like scripts/asr_eval.py.

Two resampling units are reported:
* clip     – resample clips (standard, but treats clips as independent);
* cluster  – resample whole groups of related clips: speakers for test_cv and test_parl, sentences
             for test_fleurs (FLEURS has no speaker IDs and each sentence is read by several
             speakers). This is the more honest interval, because clips of one speaker or one
             sentence are not independent (Week 9: related recordings).

Differences are paired: the same resampled clips are used for both systems.
Run from the repository root:  python scripts/bootstrap_ci.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr_eval as E  # noqa: E402

B = 2000
SEED = 0
OUT_CSV = "results/bootstrap_ci.csv"
OUT_MD = "results/bootstrap_ci.md"


def per_clip_counts(df):
    """Word and character edit counts and reference lengths per clip."""
    import jiwer

    we, wn, ce, cn = [], [], [], []
    for r, h in zip(df.reference, df.hypothesis):
        w = jiwer.process_words(r, h)
        c = jiwer.process_characters(r, h)
        we.append(w.substitutions + w.deletions + w.insertions)
        wn.append(w.hits + w.substitutions + w.deletions)
        ce.append(c.substitutions + c.deletions + c.insertions)
        cn.append(c.hits + c.substitutions + c.deletions)
    return np.array(we), np.array(wn), np.array(ce), np.array(cn)


def cluster_ids(split):
    d = pd.read_csv(f"data/{split}.csv", keep_default_na=False, dtype={"speaker": str})
    key = d.text if split == "test_fleurs" else d.speaker      # FLEURS: no speaker ids -> sentence
    return dict(zip(d.id, key))


def resample_index(groups, rng):
    """Indices of a cluster bootstrap sample (draw whole groups with replacement)."""
    keys = list(groups)
    pick = rng.integers(0, len(keys), len(keys))
    return np.concatenate([groups[keys[i]] for i in pick])


def main():
    systems = E.load_scores()[["system", "role"]].drop_duplicates()
    role = dict(zip(systems.system, systems.role))
    slug2name = {E._slug(s): s for s in systems.system}

    rows = []
    for split in E.TEST_SETS:
        files = sorted(glob.glob(f"{E.PRED_DIR}/*__{split}.csv"))
        data = {}
        for f in files:
            slug = os.path.basename(f).split("__")[0]
            if slug not in slug2name:
                continue
            df = pd.read_csv(f, keep_default_na=False, dtype=str)
            data[slug2name[slug]] = (df.id.tolist(), per_clip_counts(df))
        if not data:
            continue
        ids = next(iter(data.values()))[0]
        assert all(v[0] == ids for v in data.values()), f"{split}: systems scored on different clips"
        cl = cluster_ids(split)
        groups = {}
        for i, cid in enumerate(ids):
            groups.setdefault(cl[cid], []).append(i)
        groups = {k: np.array(v) for k, v in groups.items()}

        rng = np.random.default_rng(SEED)
        samples = {"clip": [rng.integers(0, len(ids), len(ids)) for _ in range(B)],
                   "cluster": [resample_index(groups, rng) for _ in range(B)]}

        names = list(data)
        stats = {}
        for n in names:
            we, wn, ce, cn = data[n][1]
            stats[n] = {"wer": 100 * we.sum() / wn.sum(), "cer": 100 * ce.sum() / cn.sum()}
            for unit, idx in samples.items():
                stats[n][unit] = (np.array([100 * we[i].sum() / wn[i].sum() for i in idx]),
                                  np.array([100 * ce[i].sum() / cn[i].sum() for i in idx]))

        def ci(a):
            return np.percentile(a, 2.5), np.percentile(a, 97.5)

        for n in names:
            row = {"test_set": split, "comparison": n, "kind": "system", "n_clips": len(ids),
                   "n_clusters": len(groups), "wer": round(stats[n]["wer"], 2), "cer": round(stats[n]["cer"], 2)}
            for unit in samples:
                lo, hi = ci(stats[n][unit][0])
                row[f"wer_ci_{unit}"] = f"[{lo:.2f}, {hi:.2f}]"
                lo, hi = ci(stats[n][unit][1])
                row[f"cer_ci_{unit}"] = f"[{lo:.2f}, {hi:.2f}]"
            rows.append(row)

        # paired differences: every contribution vs every baseline, and baselines among themselves
        pairs = [(a, b) for a in names for b in names if a != b and
                 ((role[a] == "contribution" and role[b] == "baseline") or
                  (role[a] == role[b] == "baseline" and a < b))]
        for a, b in pairs:
            row = {"test_set": split, "comparison": f"{a} − {b}", "kind": "difference",
                   "n_clips": len(ids), "n_clusters": len(groups),
                   "wer": round(stats[a]["wer"] - stats[b]["wer"], 2),
                   "cer": round(stats[a]["cer"] - stats[b]["cer"], 2)}
            for unit in samples:
                for m, k in (("wer", 0), ("cer", 1)):
                    diff = stats[a][unit][k] - stats[b][unit][k]
                    lo, hi = ci(diff)
                    sig = "" if lo <= 0 <= hi else " *"
                    row[f"{m}_ci_{unit}"] = f"[{lo:.2f}, {hi:.2f}]{sig}"
            rows.append(row)

    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    md = ["# Bootstrap 95% confidence intervals", "",
          f"Generated by `scripts/bootstrap_ci.py` ({B} resamples, seed {SEED}) from `results/predictions/`.",
          "WER/CER in %. *Cluster* resamples whole speakers (test_cv, test_parl) or whole sentences "
          "(test_fleurs, no speaker IDs); *clip* resamples single clips. For differences, negative = the first "
          "system is better, and `*` marks an interval that excludes 0.", ""]
    for kind, title in (("system", "Systems"), ("difference", "Paired differences (A − B)")):
        md += [f"## {title}", "",
               "| Test set | " + ("System" if kind == "system" else "A − B") +
               " | WER | 95% CI (cluster) | 95% CI (clip) | CER | 95% CI (cluster) | 95% CI (clip) |",
               "|---|---|---:|---|---|---:|---|---|"]
        for r in out[out.kind == kind].itertuples():
            md.append(f"| {r.test_set} | {r.comparison} | {r.wer:.2f} | {r.wer_ci_cluster} | {r.wer_ci_clip} | "
                      f"{r.cer:.2f} | {r.cer_ci_cluster} | {r.cer_ci_clip} |")
        md.append("")
    md += ["Clusters: test_cv " + str(out[out.test_set == 'test_cv'].n_clusters.max()) + " speakers, test_parl "
           + str(out[out.test_set == 'test_parl'].n_clusters.max()) + " speakers, test_fleurs "
           + str(out[out.test_set == 'test_fleurs'].n_clusters.max()) + " sentences."]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
