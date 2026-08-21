#!/usr/bin/env python3
"""Reproduce the empirical search-scale analysis.

The script downloads two pinned configurations from the public Monkey Business
dataset when needed, verifies their SHA-256 digests, and analyzes 2.54 million
objectively graded GSM8K solutions.  A disjoint 2,000-solution reference split
per problem estimates answer-consensus scores.  The remaining 8,000 solutions
form the candidate population.  At every search breadth n, the selected-truth
rate is evaluated exactly under sampling with replacement by integrating over
the empirical order-statistic weights; no Monte Carlo search simulation is
needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "figures"

REVISION = "a9f8f73bcd6948a57ed922cba4e48062ef95f553"
FILES = {
    "Llama-3-8B-Instruct": (
        "GSM8K_Llama-3-8B-Instruct.json",
        "0e334c7010a2eb39ab0aaf38cdd196da0b6219a95fd69f82d35b8f51e46ed765",
    ),
    "Llama-3-70B-Instruct": (
        "GSM8K_Llama-3-70B-Instruct.json",
        "22ff8a363d5a5b8ea5a6c299285d8f615ec32b9c1fe08e8253cc1a6c9c7c323f",
    ),
}

N_REFERENCE = 2_000
N_DEPLOYMENT = 8_000
SEARCH_SCALES = np.array(
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096],
    dtype=int,
)
HIGH_SCALE = int(SEARCH_SCALES[-1])
N_BOOTSTRAP = 20_000
SPLIT_SEED = 20_260_817
BOOTSTRAP_SEED = 73_041
ANSWER_RE = re.compile(r"####\s*(-?[0-9.,]+)")

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B3294"
GRAY = "#666666"
LIGHT_BLUE = "#DDEBF5"
LIGHT_ORANGE = "#F7E2D9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_if_needed(path: Path, expected_sha: str) -> None:
    if path.exists() and sha256(path) == expected_sha:
        return
    RAW.mkdir(parents=True, exist_ok=True)
    url = (
        "https://huggingface.co/datasets/ScalingIntelligence/"
        f"monkey_business/resolve/{REVISION}/{path.name}?download=true"
    )
    temporary = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {path.name} ...")
    urllib.request.urlretrieve(url, temporary)
    observed = sha256(temporary)
    if observed != expected_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: {observed} != {expected_sha}"
        )
    temporary.replace(path)


def normalize_answer(sample: str) -> str | None:
    matches = ANSWER_RE.findall(sample)
    if not matches:
        return None
    return re.sub(r"[,\s]", "", matches[-1]).strip().lower()


def exact_winner_curve(scores: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact search curve and tie-averaged rank truth.

    Candidates are sampled iid from the empirical deployment distribution.  The
    highest score wins and ties at the winning score are broken uniformly.  A
    score group with cumulative masses F_- and F therefore wins with probability
    F**n - F_-**n, and its conditional truth rate is the group's mean label.
    The second return value spreads that group mean across all tied rank
    positions, avoiding an arbitrary fixed tie order in rank-bin plots.
    """
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    population = len(sorted_truth)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], population]
    group_truth = np.array(
        [sorted_truth[start:end].mean() for start, end in zip(starts, ends)]
    )
    upper = ends.astype(float) / population
    lower = starts.astype(float) / population
    curve = np.array(
        [np.sum(group_truth * (upper**n - lower**n)) for n in SEARCH_SCALES]
    )
    tie_averaged_truth = np.empty(population, dtype=float)
    for start, end, mean in zip(starts, ends, group_truth):
        tie_averaged_truth[start:end] = mean
    return curve, tie_averaged_truth


def analyze_model(model_index: int, model: str, records: list[dict]) -> dict:
    curves: list[np.ndarray] = []
    effects: list[dict] = []
    rank_truth = np.zeros(20, dtype=float)
    rank_count = np.zeros(20, dtype=int)

    for problem_index, record in enumerate(records):
        samples = record["samples"]
        labels = np.asarray(record["is_corrects"], dtype=float)
        if len(samples) != N_REFERENCE + N_DEPLOYMENT or len(labels) != len(samples):
            raise ValueError(
                f"Unexpected candidate count for {model}, problem {problem_index}"
            )

        split_rng = np.random.default_rng(
            SPLIT_SEED + 10_000 * model_index + problem_index
        )
        indices = split_rng.permutation(len(samples))
        reference_indices = indices[:N_REFERENCE]
        deployment_indices = indices[N_REFERENCE:]

        reference_answers = [normalize_answer(samples[i]) for i in reference_indices]
        counts = Counter(answer for answer in reference_answers if answer is not None)

        deployment_answers = [normalize_answer(samples[i]) for i in deployment_indices]
        scores = np.array(
            [
                counts.get(answer, 0) / N_REFERENCE
                if answer is not None
                else -1.0 / N_REFERENCE
                for answer in deployment_answers
            ],
            dtype=float,
        )
        truth = labels[deployment_indices]

        curve, tie_averaged_truth = exact_winner_curve(scores, truth)
        curves.append(curve)

        if len(np.unique(truth)) == 2:
            auc = float(roc_auc_score(truth, scores))
        else:
            auc = float("nan")

        effects.append(
            {
                "model": model,
                "problem_index": problem_index,
                "gsm8k_index": int(record["orig_dset_idx"]),
                "baseline_truth": float(curve[0]),
                "high_scale_truth": float(curve[-1]),
                "delta": float(curve[-1] - curve[0]),
                "within_problem_auc": auc,
            }
        )

        ranks = (np.arange(N_DEPLOYMENT) + 0.5) / N_DEPLOYMENT
        bins = np.minimum((20 * ranks).astype(int), 19)
        rank_truth += np.bincount(bins, weights=tie_averaged_truth, minlength=20)
        rank_count += np.bincount(bins, minlength=20)

    curve_matrix = np.vstack(curves)
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED + model_index)
    bootstrap_indices = bootstrap_rng.integers(
        0, len(records), size=(N_BOOTSTRAP, len(records))
    )
    bootstrap_means = curve_matrix[bootstrap_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975], axis=0)
    bootstrap_deltas = bootstrap_means[:, -1] - bootstrap_means[:, 0]
    delta_lower, delta_upper = np.quantile(bootstrap_deltas, [0.025, 0.975])

    macro_auc = np.array([row["within_problem_auc"] for row in effects], dtype=float)
    bootstrap_auc = np.nanmean(macro_auc[bootstrap_indices], axis=1)
    auc_lower, auc_upper = np.quantile(bootstrap_auc, [0.025, 0.975])
    deltas = curve_matrix[:, -1] - curve_matrix[:, 0]

    return {
        "model": model,
        "curves": curve_matrix,
        "mean": curve_matrix.mean(axis=0),
        "lower": lower,
        "upper": upper,
        "effects": effects,
        "rank_truth": rank_truth / rank_count,
        "summary": {
            "problems": len(records),
            "candidates_total": len(records) * (N_REFERENCE + N_DEPLOYMENT),
            "candidates_reference": len(records) * N_REFERENCE,
            "candidates_deployment": len(records) * N_DEPLOYMENT,
            "truth_n1": float(curve_matrix[:, 0].mean()),
            f"truth_n{HIGH_SCALE}": float(curve_matrix[:, -1].mean()),
            "mean_delta": float(deltas.mean()),
            "mean_delta_ci95": [float(delta_lower), float(delta_upper)],
            "macro_auc": float(np.nanmean(macro_auc)),
            "macro_auc_ci95": [float(auc_lower), float(auc_upper)],
            "problems_harmed_any": int(np.sum(deltas < 0)),
            "problems_harmed_gt_1pp": int(np.sum(deltas < -0.01)),
            "problems_harmed_gt_5pp": int(np.sum(deltas < -0.05)),
            "worst_delta": float(deltas.min()),
            "best_delta": float(deltas.max()),
        },
    }


def write_csv(results: list[dict]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    spectrum_path = PROCESSED / "search_spectrum.csv"
    with spectrum_path.open("w", encoding="utf-8") as stream:
        stream.write("model,n,mean,ci_low,ci_high\n")
        for result in results:
            for n, mean, low, high in zip(
                SEARCH_SCALES,
                result["mean"],
                result["lower"],
                result["upper"],
            ):
                stream.write(f"{result['model']},{n},{mean:.12g},{low:.12g},{high:.12g}\n")

    effect_path = PROCESSED / "problem_effects.csv"
    fields = [
        "model",
        "problem_index",
        "gsm8k_index",
        "baseline_truth",
        "high_scale_truth",
        "delta",
        "within_problem_auc",
    ]
    with effect_path.open("w", encoding="utf-8") as stream:
        stream.write(",".join(fields) + "\n")
        for result in results:
            for row in result["effects"]:
                stream.write(
                    ",".join(
                        "nan" if isinstance(row[field], float) and np.isnan(row[field])
                        else str(row[field])
                        for field in fields
                    )
                    + "\n"
                )

    rank_path = PROCESSED / "rank_conditional_truth.csv"
    with rank_path.open("w", encoding="utf-8") as stream:
        stream.write("model,rank_midpoint,truth_rate\n")
        for result in results:
            for index, rate in enumerate(result["rank_truth"]):
                stream.write(f"{result['model']},{(index + 0.5) / 20:.3f},{rate:.12g}\n")

    summary = {
        "dataset": "ScalingIntelligence/monkey_business",
        "revision": REVISION,
        "reference_candidates_per_problem": N_REFERENCE,
        "deployment_candidates_per_problem": N_DEPLOYMENT,
        "high_search_scale": HIGH_SCALE,
        "bootstrap_replicates": N_BOOTSTRAP,
        "models": {result["model"]: result["summary"] for result in results},
    }
    (PROCESSED / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def make_figure(results: list[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )

    colors = [ORANGE, BLUE]
    fills = [LIGHT_ORANGE, LIGHT_BLUE]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), constrained_layout=True)

    ax = axes[0]
    for result, color, fill in zip(results, colors, fills):
        label = result["model"].replace("-Instruct", "")
        ax.fill_between(SEARCH_SCALES, result["lower"], result["upper"], color=fill)
        ax.plot(SEARCH_SCALES, result["mean"], color=color, lw=2.0, label=label)
    ax.set(
        xscale="log",
        xlim=(1, HIGH_SCALE),
        ylim=(0.65, 1.01),
        xlabel="Candidates searched, $n$",
        ylabel="Selected-answer truth rate",
        title="A  Search changes reliability",
    )
    ax.set_xticks([1, 8, 64, 512, 4096])
    ax.set_xticklabels(["1", "8", "64", "512", "4096"])
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for result, color in zip(results, colors):
        deltas = np.sort(result["curves"][:, -1] - result["curves"][:, 0])
        quantiles = (np.arange(len(deltas)) + 0.5) / len(deltas)
        label = result["model"].replace("-Instruct", "")
        ax.plot(quantiles, deltas, lw=1.8, color=color, label=label)
    ax.axhline(0, color=GRAY, ls="--", lw=1.0)
    ax.set(
        xlim=(0, 1),
        ylim=(-0.55, 0.95),
        xlabel="Problem quantile",
        ylabel=f"Truth-rate change, $n=1$ to {HIGH_SCALE}",
        title="B  Average gains hide reversals",
    )
    ax.legend(frameon=False, loc="upper right")

    ax = axes[2]
    rank_midpoints = (np.arange(20) + 0.5) / 20
    for result, color in zip(results, colors):
        label = result["model"].replace("-Instruct", "")
        ax.plot(rank_midpoints, result["rank_truth"], color=color, lw=2.0, label=label)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Consensus-score percentile",
        ylabel="Candidate truth rate",
        title="C  Rank predicts truth on average",
    )
    ax.legend(frameon=False, loc="lower right")

    fig.savefig(FIG / "fig3_empirical.pdf")
    fig.savefig(FIG / "fig3_empirical.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="fail instead of downloading when a pinned raw file is absent",
    )
    args = parser.parse_args()

    results = []
    for model_index, (model, (filename, digest)) in enumerate(FILES.items()):
        path = RAW / filename
        if args.no_download:
            if not path.exists() or sha256(path) != digest:
                raise FileNotFoundError(f"Missing or invalid pinned input: {path}")
        else:
            download_if_needed(path, digest)
        print(f"Loading {filename} ...")
        with path.open(encoding="utf-8") as stream:
            records = json.load(stream)
        results.append(analyze_model(model_index, model, records))

    write_csv(results)
    make_figure(results)
    print(json.dumps({r["model"]: r["summary"] for r in results}, indent=2))


if __name__ == "__main__":
    main()
