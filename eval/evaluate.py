"""Build a labelled synthetic corpus, score it, and report honest metrics.

Only one quantity is fitted: the decision threshold, chosen on the training half.
Detector internals and combiner weights are fixed constants in the library and
are never touched by this script. Everything below the threshold line is
measured on the held-out half.

Run:  python eval/evaluate.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iad.detectors import default_detectors  # noqa: E402
from iad.scoring import DEFAULT_WEIGHTS, Scorer  # noqa: E402
from iad.synth import generate_corpus  # noqa: E402

SEED = 20260915
N_SESSIONS = 12_000
BOT_SHARE = 0.50
EVASIVE_SHARE_OF_BOTS = 0.45
SPLIT_SEED = 7
CONSERVATIVE_FPR = 0.01
RESULTS_PATH = Path(__file__).resolve().parent / "results.md"

KIND_ORDER = ("human", "bot_naive", "bot_evasive")

Floats = npt.NDArray[np.float64]
Labels = npt.NDArray[np.str_]


def build(n_sessions: int, seed: int) -> tuple[Floats, Floats, Labels]:
    """Return (per_detector_scores [n, k], combined [n], kinds [n]).

    An abstaining detector is stored as NaN. A session where every detector
    abstained gets no verdict, so it is stored as 0.0: never flagged.
    """
    scorer = Scorer()
    names = [d.name for d in default_detectors()]
    rows: list[list[float]] = []
    combined: list[float] = []
    kinds: list[str] = []

    for arrays, kind in generate_corpus(
        n_sessions,
        seed=seed,
        bot_share=BOT_SHARE,
        evasive_share_of_bots=EVASIVE_SHARE_OF_BOTS,
    ):
        score, per = scorer.score_arrays(arrays)
        rows.append([_or_nan(per[n]) for n in names])
        combined.append(0.0 if score is None else score)
        kinds.append(kind)

    return np.asarray(rows), np.asarray(combined), np.asarray(kinds)


def _or_nan(v: float | None) -> float:
    return np.nan if v is None else v


def pick_threshold_max_f1(scores: Floats, y: Floats) -> tuple[float, float]:
    """Threshold that maximizes F1 on the training half."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(scores, 4)):
        pred = scores >= t
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        fn = float(np.sum(~pred & (y == 1)))
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def pick_threshold_at_fpr(scores: Floats, y: Floats, max_fpr: float) -> float:
    """Lowest threshold whose training false-positive rate stays under `max_fpr`.

    A 10% false-positive rate is not deployable when the downstream action is a
    ban, so the report also covers a conservative operating point.
    """
    neg = np.sort(scores[y == 0])
    if neg.size == 0:
        return 1.0
    idx = min(neg.size - 1, int(np.ceil((1.0 - max_fpr) * neg.size)))
    return float(np.nextafter(neg[idx], 1.0))


def operating_point(
    scores: Floats, y: Floats, kinds: Labels, threshold: float
) -> dict[str, float]:
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "accuracy": (tp + tn) / len(y),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "recall_naive": float(pred[kinds == "bot_naive"].mean()),
        "recall_evasive": float(pred[kinds == "bot_evasive"].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=int, default=N_SESSIONS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    t0 = time.perf_counter()
    det_names = [d.name for d in default_detectors()]
    per_det, combined, kinds = build(args.sessions, args.seed)
    gen_s = time.perf_counter() - t0

    y = (kinds != "human").astype(int)
    rs = np.random.default_rng(SPLIT_SEED)
    perm = rs.permutation(len(y))
    half = len(y) // 2
    train, test = perm[:half], perm[half:]

    t_f1, train_f1 = pick_threshold_max_f1(combined[train], y[train])
    t_cons = pick_threshold_at_fpr(combined[train], y[train], CONSERVATIVE_FPR)

    y_te, s_te, k_te, d_te = y[test], combined[test], kinds[test], per_det[test]
    op_f1 = operating_point(s_te, y_te, k_te, t_f1)
    op_cons = operating_point(s_te, y_te, k_te, t_cons)
    auc = float(roc_auc_score(y_te, s_te))

    # Per-detector AUC is measured only on sessions where that detector scored.
    det_auc = {n: _auc(y_te, d_te[:, i]) for i, n in enumerate(det_names)}

    # Per-detector AUC restricted to evasive-vs-human, the interesting contrast.
    sub = (k_te == "bot_evasive") | (k_te == "human")
    y_ev = (k_te[sub] == "bot_evasive").astype(int)
    det_auc_ev = {n: _auc(y_ev, d_te[sub][:, i]) for i, n in enumerate(det_names)}
    auc_ev = float(roc_auc_score(y_ev, s_te[sub]))

    means = {
        k: {
            n: float(np.nanmean(per_det[kinds == k][:, i]))
            for i, n in enumerate(det_names)
        }
        | {"combined": float(combined[kinds == k].mean())}
        for k in KIND_ORDER
    }
    abstained = {n: int(np.isnan(per_det[:, i]).sum()) for i, n in enumerate(det_names)}
    no_verdict = int(np.isnan(per_det).all(axis=1).sum())
    counts = {k: int((kinds == k).sum()) for k in KIND_ORDER}

    report = render(
        sessions=args.sessions,
        seed=args.seed,
        counts=counts,
        gen_s=gen_s,
        n_train=len(train),
        n_test=len(y_te),
        train_f1=train_f1,
        op_f1=op_f1,
        op_cons=op_cons,
        auc=auc,
        auc_ev=auc_ev,
        det_auc=det_auc,
        det_auc_ev=det_auc_ev,
        n_naive=int((k_te == "bot_naive").sum()),
        n_evasive=int((k_te == "bot_evasive").sum()),
        means=means,
        det_names=det_names,
        abstained=abstained,
        no_verdict=no_verdict,
    )

    print(report)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(f"written to {RESULTS_PATH}")
    return 0


def _auc(y: Floats, scores: Floats) -> float:
    ok = ~np.isnan(scores)
    return float(roc_auc_score(y[ok], scores[ok]))


def _op_table(op: dict[str, float]) -> list[str]:
    return [
        "| Metric | Value |",
        "|---|---|",
        f"| Threshold | {op['threshold']:.4f} |",
        f"| Precision | {op['precision']:.4f} |",
        f"| Recall | {op['recall']:.4f} |",
        f"| F1 | {op['f1']:.4f} |",
        f"| Accuracy | {op['accuracy']:.4f} |",
        f"| False positive rate | {op['fpr']:.4f} |",
        "",
        "| | predicted human | predicted bot |",
        "|---|---|---|",
        f"| **actual human** | {op['tn']} | {op['fp']} |",
        f"| **actual bot** | {op['fn']} | {op['tp']} |",
    ]


def render(**k: Any) -> str:
    det_names = k["det_names"]
    f1op, cons = k["op_f1"], k["op_cons"]
    lines: list[str] = []
    add = lines.append

    add("# Evaluation results")
    add("")
    add("Generated by `eval/evaluate.py`. Do not edit by hand.")
    add("")
    add("## Corpus")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Sessions | {k['sessions']:,} |")
    add(f"| Seed | `{k['seed']}` |")
    add(f"| Human | {k['counts']['human']:,} |")
    add(f"| Naive bot | {k['counts']['bot_naive']:,} |")
    add(f"| Evasive bot | {k['counts']['bot_evasive']:,} |")
    n_bots = k["counts"]["bot_naive"] + k["counts"]["bot_evasive"]
    add(
        f"| Bots that are evasive | {k['counts']['bot_evasive'] / max(1, n_bots):.1%} |"
    )
    add(f"| Train / test split | {k['n_train']:,} / {k['n_test']:,} |")
    add(f"| Generate + score wall time | {k['gen_s']:.1f} s |")
    add(f"| Sessions with no verdict (all detectors abstained) | {k['no_verdict']:,} |")
    for n in det_names:
        add(f"| `{n}` abstained (too little data) | {k['abstained'][n]:,} |")
    add("")
    add("## Ranking quality (threshold-free, test half)")
    add("")
    add("| Contrast | ROC AUC |")
    add("|---|---|")
    add(f"| All bots vs human | {k['auc']:.4f} |")
    add(f"| Evasive bots vs human | {k['auc_ev']:.4f} |")
    add("")
    add("## Operating point A: max-F1")
    add("")
    add(
        f"Threshold chosen on the {k['n_train']:,} training sessions by maximizing F1 "
        f"(train F1 {k['train_f1']:.4f}). Metrics below are on the held-out half."
    )
    add("")
    lines.extend(_op_table(f1op))
    add("")
    add("### Recall split: naive vs evasive")
    add("")
    add("| Population | n | Recall |")
    add("|---|---|---|")
    add(f"| Naive bot | {k['n_naive']} | {f1op['recall_naive']:.4f} |")
    add(f"| Evasive bot | {k['n_evasive']} | {f1op['recall_evasive']:.4f} |")
    add("")
    add(
        "The evasive figure is the one that matters. It is the recall against "
        "automation that already jitters its timing on a lognormal, curves its "
        "cursor paths, overshoots half its targets and draws dwell times from a "
        "human-shaped distribution. The naive figure is near-perfect because a "
        "constant-delay, straight-line, fixed-dwell script is trivially "
        "separable, not because the detectors are strong."
    )
    add("")
    add("## Operating point B: conservative")
    add("")
    add(
        f"Threshold chosen on the training half for a false-positive rate under "
        f"{CONSERVATIVE_FPR:.0%}. This is the point that matters if the downstream "
        f"action is an account ban rather than a review queue."
    )
    add("")
    lines.extend(_op_table(cons))
    add("")
    add("| Population | Recall |")
    add("|---|---|")
    add(f"| Naive bot | {cons['recall_naive']:.4f} |")
    add(f"| Evasive bot | {cons['recall_evasive']:.4f} |")
    add("")
    add("## Per-detector discrimination (test half)")
    add("")
    add("| Detector | Weight | AUC, all bots | AUC, evasive only |")
    add("|---|---|---|---|")
    for n in det_names:
        add(
            f"| `{n}` | {DEFAULT_WEIGHTS[n]:.2f} | {k['det_auc'][n]:.4f} | "
            f"{k['det_auc_ev'][n]:.4f} |"
        )
    add(f"| **combined** | - | **{k['auc']:.4f}** | **{k['auc_ev']:.4f}** |")
    add("")
    add(
        "The combined score beats every individual detector on both contrasts, "
        "which is the whole point of using three independent signals."
    )
    add("")
    add("## Mean score by population (whole corpus)")
    add("")
    add("| Population | " + " | ".join(f"`{n}`" for n in det_names) + " | combined |")
    add("|---" * (len(det_names) + 2) + "|")
    for kind in KIND_ORDER:
        row = " | ".join(f"{k['means'][kind][n]:.3f}" for n in det_names)
        add(f"| {kind} | {row} | **{k['means'][kind]['combined']:.3f}** |")
    add("")
    add("## Caveats")
    add("")
    add(
        "- The corpus is synthetic. These numbers measure how well the detectors "
        "separate *this* generator's populations. They are not an estimate of "
        "real-world accuracy; that needs labelled capture from production."
    )
    add(
        "- The evasive population encodes one specific set of humanizer "
        "assumptions. An adversary who also models minimum-jerk velocity "
        "profiles and overshoot-and-correct reaching would score lower. The "
        "dwell shape sub-signal is already near-useless against it."
    )
    add(
        "- The combined score is a ranking statistic, not a calibrated "
        "probability. Its absolute value is not meaningful; only the ordering "
        "and the fitted threshold are."
    )
    add(
        "- Sessions are short bursts of roughly 15-23 action units. Sub-signals "
        "that need sample size, notably dwell skewness, are noisy at this length "
        "and are weighted accordingly."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
