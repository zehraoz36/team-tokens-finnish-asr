"""Shared evaluation code for Team Tokens Finnish ASR (COE025 Project 3).

Every system (baselines and our fine-tuned models) is scored with this file, so
that all results use the same test clips, the same references, one text
normalization policy and the same WER/CER computation.

Policy
------
* References: the ``text`` column of the Hugging Face dataset. It is
  ``normalize(raw_text)`` produced by ``scripts/prepare_data.ipynb``.
* Hypotheses: ``normalize(raw model output)``.
* ``normalize`` is applied exactly ONCE to a raw string. Do not normalize an
  already-normalized reference again: the thousands rule would merge numbers
  ("150 200" -> "150200").
* WER and CER are corpus-level (total edits / total reference words or
  characters, spaces included for CER), computed with ``jiwer``. They are not
  averages of per-clip scores (Week 9, slide 12).
* The three test sets are always reported separately, never averaged.
* RTF = compute seconds / audio seconds. The timer covers only the
  ``transcribe_fn`` call (feature extraction + decoding + text decoding) on
  audio that is already in memory; data loading and audio decoding are
  excluded. RTFx = 1 / RTF.

Colab usage
-----------
    !git clone -q https://github.com/zehraoz36/team-tokens-finnish-asr
    import sys; sys.path.insert(0, "team-tokens-finnish-asr/scripts")
    import asr_eval as E
    test = E.load_test_sets()
    E.evaluate_system("whisper-small zero-shot", "openai/whisper-small",
                      my_transcribe_fn, test, role="baseline", settings="...")
    E.show_results()

Outputs: ``results/scores/<system>.csv`` (one row per test set) and
``results/predictions/<system>__<test_set>.csv`` (id, reference, hypothesis,
raw_hypothesis).

``transcribe_fn(list_of_float32_arrays_16k) -> list_of_raw_strings``.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import platform
import re
import time
import unicodedata
from typing import Callable, Dict, List, Optional, Sequence

HF_DATASET = "zehraoz36/team-tokens-finnish-asr-data"
TEST_SETS = ("test_cv", "test_parl", "test_fleurs")
SAMPLING_RATE = 16_000

SCORES_DIR = "results/scores"          # one CSV per system -> no merge conflicts
PRED_DIR = "results/predictions"       # one CSV per system and test set

RESULT_FIELDS = [
    "system", "role", "checkpoint", "revision", "test_set", "clips",
    "ref_words", "wer", "cer", "substitutions", "deletions", "insertions",
    "audio_hours", "compute_sec", "rtf", "rtfx", "hardware", "precision",
    "batch_size", "settings", "date",
]

# --------------------------------------------------------------------------
# 1. Text normalization (identical to scripts/prepare_data.ipynb, section 2)
# --------------------------------------------------------------------------
_BRACKETS = re.compile(r"\[[^\]]*\]|<[^>]*>|\(\S*\)\s*$")   # [noise], <unk>, trailing (utt-id)


def normalize(text) -> str:
    """Lowercase, punctuation -> space, keep Finnish letters and digits.

    Apply once to a raw string (reference ``raw_text`` or raw model output).
    """
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _BRACKETS.sub(" ", text)
    text = text.lower()
    text = re.sub(r"(?<=\d)[ .,](?=\d{3}\b)", "", text)   # 1 500 / 1.500 -> 1500
    text = re.sub(r"[^\w\s]", " ", text)                     # punctuation (also - and ') -> space
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# 2. Scoring
# --------------------------------------------------------------------------
def score(refs: Sequence[str], hyps: Sequence[str]) -> Dict[str, float]:
    """Corpus-level WER/CER (in %) plus word edit counts.

    ``refs`` and ``hyps`` must already be normalized.
    """
    import jiwer

    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    if any(not r for r in refs):
        raise ValueError("empty reference found; the test sets should not contain any")
    w = jiwer.process_words(list(refs), list(hyps))
    c = jiwer.process_characters(list(refs), list(hyps))
    n_words = w.hits + w.substitutions + w.deletions
    return {
        "wer": round(100 * w.wer, 2),
        "cer": round(100 * c.cer, 2),
        "substitutions": w.substitutions,
        "deletions": w.deletions,
        "insertions": w.insertions,
        "ref_words": n_words,
    }


def score_file(path: str) -> Dict[str, float]:
    """Re-score a saved prediction file (columns ``reference``, ``hypothesis``)."""
    import pandas as pd

    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    return score(df["reference"].tolist(), df["hypothesis"].tolist())


# --------------------------------------------------------------------------
# 3. Data
# --------------------------------------------------------------------------
def load_test_sets(splits: Sequence[str] = TEST_SETS, repo: str = HF_DATASET):
    """Load the test splits from Hugging Face, audio decoded at 16 kHz mono."""
    from datasets import Audio, load_dataset

    out = {}
    for s in splits:
        ds = load_dataset(repo, split=s)
        out[s] = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    return out


# --------------------------------------------------------------------------
# 4. Bookkeeping: revision, hardware, files
# --------------------------------------------------------------------------
def model_revision(model_id: str) -> str:
    """Commit hash of a Hugging Face model (``local`` if it is not on the Hub)."""
    if os.path.isdir(model_id):
        return "local"
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha
    except Exception as e:  # offline or private
        return f"unknown ({type(e).__name__})"


def hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)} (torch {torch.__version__})"
        return f"CPU {platform.processor() or platform.machine()} (torch {torch.__version__})"
    except ImportError:
        return f"CPU {platform.machine()}"


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def save_predictions(path: str, ids, refs, raw_hyps, hyps) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "reference", "hypothesis", "raw_hypothesis"])
        for row in zip(ids, refs, hyps, raw_hyps):
            w.writerow(row)


def append_result(row: Dict, scores_dir: str = SCORES_DIR) -> None:
    """Add one row to the system's score file, replacing an older row of the
    same (system, test_set)."""
    path = f"{scores_dir}/{_slug(row['system'])}.csv"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if (r["system"], r["test_set"]) != (row["system"], row["test_set"])]
    rows.append({k: row.get(k, "") for k in RESULT_FIELDS})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# 5. Run one system on the test sets
# --------------------------------------------------------------------------
def run_split(transcribe_fn: Callable[[List], List[str]], dset, batch_size: int = 16,
              desc: str = ""):
    """Transcribe one split. Returns ids, refs, raw_hyps, compute_sec, audio_sec."""
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731

    ids, refs, raw = [], [], []
    compute = 0.0
    audio_sec = 0.0
    for i in tqdm(range(0, len(dset), batch_size), desc=desc):
        batch = dset[i:i + batch_size]
        arrays = [a["array"] for a in batch["audio"]]
        audio_sec += sum(len(a) for a in arrays) / SAMPLING_RATE
        _sync()
        t0 = time.perf_counter()
        out = transcribe_fn(arrays)
        _sync()
        compute += time.perf_counter() - t0
        if len(out) != len(arrays):
            raise RuntimeError(f"transcribe_fn returned {len(out)} outputs for {len(arrays)} clips")
        ids += list(batch["id"])
        refs += list(batch["text"])          # already normalize(raw_text)
        raw += [str(o) if o is not None else "" for o in out]
    return ids, refs, raw, compute, audio_sec


def evaluate_system(system: str, checkpoint: str, transcribe_fn, test_sets: Dict,
                    role: str = "baseline", settings: str = "", precision: str = "",
                    batch_size: int = 16, revision: Optional[str] = None,
                    scores_dir: str = SCORES_DIR, pred_dir: str = PRED_DIR) -> List[Dict]:
    """Transcribe every test set, save predictions and append rows to the results table.

    role: "baseline" (pretrained, no training by us) or "contribution" (trained by us).
    """
    if role not in ("baseline", "contribution"):
        raise ValueError("role must be 'baseline' or 'contribution'")
    revision = revision or model_revision(checkpoint)
    hw = hardware()
    rows = []
    for split, dset in test_sets.items():
        ids, refs, raw, compute, audio_sec = run_split(transcribe_fn, dset, batch_size, desc=f"{system} {split}")
        hyps = [normalize(h) for h in raw]
        save_predictions(f"{pred_dir}/{_slug(system)}__{split}.csv", ids, refs, raw, hyps)
        s = score(refs, hyps)
        rtf = compute / audio_sec if audio_sec else float("nan")
        row = dict(system=system, role=role, checkpoint=checkpoint, revision=revision,
                   test_set=split, clips=len(ids), audio_hours=round(audio_sec / 3600, 3),
                   compute_sec=round(compute, 1), rtf=round(rtf, 4),
                   rtfx=round(1 / rtf, 1) if rtf else "", hardware=hw, precision=precision,
                   batch_size=batch_size, settings=settings,
                   date=_dt.date.today().isoformat(), **s)
        append_result(row, scores_dir)
        rows.append(row)
        print(f"[{system}] {split:12s} WER={s['wer']:6.2f}  CER={s['cer']:6.2f}  RTF={rtf:.4f}")
    return rows


def load_scores(scores_dir: str = SCORES_DIR):
    """All systems' score rows in one DataFrame."""
    import glob

    import pandas as pd

    files = sorted(glob.glob(f"{scores_dir}/*.csv"))
    if not files:
        raise FileNotFoundError(f"no score files in {scores_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def show_results(scores_dir: str = SCORES_DIR):
    """Results as a table: one row per system, WER/CER per test set."""
    df = load_scores(scores_dir)
    t = df.pivot_table(index=["role", "system"], columns="test_set", values=["wer", "cer"])
    order = [(m, s) for m in ("wer", "cer") for s in TEST_SETS if (m, s) in t.columns]
    return t[order].round(2)


def export_zip(systems: Sequence[str], zip_path: str) -> List[str]:
    """Zip the score file and prediction files of the given systems (for download from Colab)."""
    import glob
    import zipfile

    files = []
    for name in systems:
        slug = _slug(name)
        files += glob.glob(f"{SCORES_DIR}/{slug}.csv") + sorted(glob.glob(f"{PRED_DIR}/{slug}__*.csv"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f)
    print(f"{zip_path}: {len(files)} files")
    return files
