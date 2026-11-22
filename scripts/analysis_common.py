"""Read and validate saved ASR predictions without loading audio or models."""
import csv
import hashlib
from pathlib import Path

import jiwer
import numpy as np

import asr_eval as E

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def unique_rows(rows, source):
    result = {}
    for r in rows:
        if not r["id"] or r["id"] in result:
            raise ValueError(f"{source}: missing or duplicate id {r['id']!r}")
        result[r["id"]] = r
    return result


def counts(reference, hypothesis):
    w = jiwer.process_words(reference, hypothesis)
    c = jiwer.process_characters(reference, hypothesis)
    return (w.substitutions + w.deletions + w.insertions,
            w.hits + w.substitutions + w.deletions,
            c.substitutions + c.deletions + c.insertions,
            c.hits + c.substitutions + c.deletions,
            w.substitutions, w.deletions, w.insertions)


def align_predictions(manifest, predictions, source):
    indexed = unique_rows(predictions, source)
    expected = unique_rows(manifest, "manifest")
    if set(indexed) != set(expected):
        raise ValueError(f"{source}: prediction IDs do not match full manifest")
    ordered = [indexed[r["id"]] for r in manifest]
    for r, p in zip(manifest, ordered):
        if not r["text"] or p["reference"] != r["text"]:
            raise ValueError(f"{source}: reference mismatch for {r['id']}")
        if E.normalize(p["raw_hypothesis"]) != p["hypothesis"]:
            raise ValueError(f"{source}: hypothesis normalization mismatch for {r['id']}")
    return ordered


def load_verified(root=ROOT):
    manifests = {s: read_csv(root / f"data/{s}.csv") for s in E.TEST_SETS}
    data, files = {}, []
    for path in sorted((root / "results/scores").glob("*.csv")):
        score_rows = read_csv(path)
        if len(score_rows) != 3 or {r["test_set"] for r in score_rows} != set(E.TEST_SETS):
            raise ValueError(f"{path}: expected one score per full test set")
        files.append(path)
        for score in score_rows:
            split, system = score["test_set"], score["system"]
            key = (system, split)
            if key in data:
                raise ValueError(f"duplicate system/split: {key}")
            pred_path = root / f"results/predictions/{E._slug(system)}__{split}.csv"
            pred = align_predictions(manifests[split], read_csv(pred_path), pred_path)
            a = np.asarray([counts(p["reference"], p["hypothesis"]) for p in pred], dtype=np.int64)
            totals = a.sum(axis=0)
            checks = {"clips": len(pred), "ref_words": int(totals[1]),
                      "substitutions": int(totals[4]), "deletions": int(totals[5]),
                      "insertions": int(totals[6]),
                      "wer": round(100 * totals[0] / totals[1], 2),
                      "cer": round(100 * totals[2] / totals[3], 2)}
            for field, value in checks.items():
                if abs(float(score[field]) - value) > 1e-8:
                    raise ValueError(f"{path}/{split}: {field} mismatch {value} != {score[field]}")
            data[key] = {"predictions": pred, "counts": a, "score": score}
            files.append(pred_path)
    actual = set((root / "results/predictions").glob("*.csv"))
    indexed = {p for p in files if p.parent.name == "predictions"}
    if actual != indexed:
        raise ValueError("Orphan prediction files not represented by score files")
    files += [root / f"data/{s}.csv" for s in E.TEST_SETS]
    return manifests, data, files


def hashes(files, root=ROOT):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files))}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
