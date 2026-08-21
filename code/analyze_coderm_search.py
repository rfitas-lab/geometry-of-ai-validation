#!/usr/bin/env python3
"""Reconstruct the independent CodeRM/HumanEval+ search experiment.

The public CodeRM release contains 100 generated unit tests for each of 100
candidate programs on every HumanEval+ task.  This script turns the resulting
candidate-by-test execution matrix into a verifier score (fraction of generated
tests passed), joins the objective HumanEval+ ``plus_status`` label, and
computes exact best-of-n reliability curves with uniform tie breaking.

Raw programs and execution logs are intentionally not copied into the paper's
reproducibility package.  The package contains the derived candidate-level
numeric table, from which the lightweight reproduction regenerates every
reported CodeRM result and figure panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

CODERM_COMMIT = "aa4946e9245ed41e24d60ad29e965132b5b84fe6"
ARCHIVE_SHA256 = "728780d642f465e91f8619de1bae0cb3192519278e87014d546cee9921d4e6c5"
N_SOLUTIONS = 100
N_TESTS = 100
SEARCH_SCALES = np.array([1, 2, 4, 8, 16, 32, 64, 100], dtype=int)
N_BOOTSTRAP = 20_000
BOOTSTRAP_SEED = 92_771

MODELS = {
    "Llama-3-8B": {
        "annotations": "sol_llama3-8b_200_anno.jsonl",
        "annotations_sha256": "cae513c7578a52c6d3110a13bbf2514e44a0cd58f2a4a258b58e7337c2af0211",
        "results_member": "llama3-8b_sol_coderm-8b_ut/details/100_sol_100_ut_result.jsonl",
        "results_sha256": "8169be74c8144e0bad6b593093a02e4963144f66a6a5f6b84c10ff1dd6e316d1",
    },
    "Llama-3-70B": {
        "annotations": "sol_llama3-70b_200_anno.jsonl",
        "annotations_sha256": "6ddb4273e4d9a35f7bef6016f37ac2e9b99d6bd7aad05357f7ff9399b0d45850",
        "results_member": "llama3-70b_sol_coderm-8b_ut/details/100_sol_100_ut_result.jsonl",
        "results_sha256": "7edb5fc98033f5de748c5dfe2f740cdeffb02020ead8f49988ea822d2d8d2fdf",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"Checksum mismatch for {path}: {observed} != {expected}")


def exact_winner_curve(
    scores: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact iid-with-replacement winner curve and tie-averaged rank truth."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    population = len(scores)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], population]
    group_truth = np.array(
        [sorted_truth[start:end].mean() for start, end in zip(starts, ends)]
    )
    lower = starts / population
    upper = ends / population
    curve = np.array(
        [np.sum(group_truth * (upper**n - lower**n)) for n in SEARCH_SCALES]
    )
    tie_truth = np.empty(population, dtype=float)
    for start, end, value in zip(starts, ends, group_truth):
        tie_truth[start:end] = value
    return curve, tie_truth


def auc_with_ties(scores: np.ndarray, truth: np.ndarray) -> float:
    """Pairwise AUC with half credit for tied scores."""
    positive = scores[truth == 1]
    negative = scores[truth == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def load_truth(path: Path) -> tuple[list[str], np.ndarray]:
    task_ids: list[str] = []
    rows: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            solutions = record["solutions"]
            if len(solutions) < N_SOLUTIONS:
                raise ValueError(f"Fewer than {N_SOLUTIONS} solutions for {record['task_id']}")
            task_ids.append(record["task_id"])
            rows.append(
                np.array(
                    [entry["plus_status"] == "pass" for entry in solutions[:N_SOLUTIONS]],
                    dtype=np.int8,
                )
            )
    return task_ids, np.vstack(rows)


def load_scores(path: Path, task_ids: list[str]) -> np.ndarray:
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    passed = np.zeros((len(task_ids), N_SOLUTIONS), dtype=np.int16)
    observed = np.zeros_like(passed)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            try:
                row = task_index[record["task_id"]]
            except KeyError as error:
                raise ValueError(f"Unknown task at line {line_number}") from error
            column = int(record["sol_id"])
            if not 0 <= column < N_SOLUTIONS:
                raise ValueError(f"Unexpected solution id at line {line_number}")
            observed[row, column] += 1
            passed[row, column] += record["result"] == "pass"
    if not np.all(observed == N_TESTS):
        bad = np.argwhere(observed != N_TESTS)[0]
        raise ValueError(
            f"Expected {N_TESTS} tests per candidate; first mismatch {tuple(bad)} has {observed[tuple(bad)]}"
        )
    return passed.astype(float) / N_TESTS


def analyze_model(model_index: int, model: str, task_ids: list[str], scores: np.ndarray, truth: np.ndarray) -> dict:
    curves: list[np.ndarray] = []
    effects: list[dict] = []
    rank_sum = np.zeros(20, dtype=float)
    rank_count = np.zeros(20, dtype=int)

    for task_index, task_id in enumerate(task_ids):
        curve, tie_truth = exact_winner_curve(scores[task_index], truth[task_index])
        curves.append(curve)
        auc = auc_with_ties(scores[task_index], truth[task_index])
        effects.append(
            {
                "model": model,
                "task_id": task_id,
                "truth_n1": float(curve[0]),
                "truth_n100": float(curve[-1]),
                "delta": float(curve[-1] - curve[0]),
                "within_task_auc": auc,
            }
        )
        bins = np.minimum((20 * (np.arange(N_SOLUTIONS) + 0.5) / N_SOLUTIONS).astype(int), 19)
        rank_sum += np.bincount(bins, weights=tie_truth, minlength=20)
        rank_count += np.bincount(bins, minlength=20)

    curve_matrix = np.vstack(curves)
    rng = np.random.default_rng(BOOTSTRAP_SEED + model_index)
    bootstrap_indices = rng.integers(0, len(task_ids), size=(N_BOOTSTRAP, len(task_ids)))
    bootstrap_means = curve_matrix[bootstrap_indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975], axis=0)
    aucs = np.array([row["within_task_auc"] for row in effects])
    bootstrap_auc = np.nanmean(aucs[bootstrap_indices], axis=1)
    auc_low, auc_high = np.quantile(bootstrap_auc, [0.025, 0.975])
    deltas = curve_matrix[:, -1] - curve_matrix[:, 0]
    delta_boot = bootstrap_means[:, -1] - bootstrap_means[:, 0]
    delta_low, delta_high = np.quantile(delta_boot, [0.025, 0.975])

    return {
        "model": model,
        "curves": curve_matrix,
        "mean": curve_matrix.mean(axis=0),
        "low": low,
        "high": high,
        "effects": effects,
        "rank_truth": rank_sum / rank_count,
        "summary": {
            "tasks": len(task_ids),
            "candidates": int(scores.size),
            "candidate_test_executions": int(scores.size * N_TESTS),
            "truth_n1": float(curve_matrix[:, 0].mean()),
            "truth_n100": float(curve_matrix[:, -1].mean()),
            "mean_delta": float(deltas.mean()),
            "mean_delta_ci95": [float(delta_low), float(delta_high)],
            "macro_auc": float(np.nanmean(aucs)),
            "macro_auc_ci95": [float(auc_low), float(auc_high)],
            "tasks_harmed_any": int(np.sum(deltas < 0)),
            "tasks_harmed_gt_1pp": int(np.sum(deltas < -0.01)),
            "tasks_harmed_gt_5pp": int(np.sum(deltas < -0.05)),
            "worst_delta": float(deltas.min()),
            "best_delta": float(deltas.max()),
        },
    }


def write_outputs(results: list[dict], candidate_rows: list[dict]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with (PROCESSED / "coderm_candidate_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "task_id", "sol_id", "verifier_score", "truth"])
        writer.writeheader()
        writer.writerows(candidate_rows)

    with (PROCESSED / "coderm_search_spectrum.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "n", "mean", "ci_low", "ci_high"])
        writer.writeheader()
        for result in results:
            for n, mean, low, high in zip(SEARCH_SCALES, result["mean"], result["low"], result["high"]):
                writer.writerow({"model": result["model"], "n": n, "mean": f"{mean:.12g}", "ci_low": f"{low:.12g}", "ci_high": f"{high:.12g}"})

    with (PROCESSED / "coderm_problem_effects.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["model", "task_id", "truth_n1", "truth_n100", "delta", "within_task_auc"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerows(result["effects"])

    with (PROCESSED / "coderm_rank_conditional_truth.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "rank_bin", "rank_midpoint", "truth_rate"])
        writer.writeheader()
        for result in results:
            for index, value in enumerate(result["rank_truth"]):
                writer.writerow({"model": result["model"], "rank_bin": index, "rank_midpoint": (index + 0.5) / 20, "truth_rate": f"{value:.12g}"})

    metadata = {
        "source": "CodeRM public repository and execution-output archive",
        "repository": "https://github.com/RUCKBReasoning/CodeRM",
        "commit": CODERM_COMMIT,
        "archive_sha256": ARCHIVE_SHA256,
        "search_scales": SEARCH_SCALES.tolist(),
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "models": {result["model"]: result["summary"] for result in results},
        "source_checksums": MODELS,
    }
    with (PROCESSED / "coderm_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", type=Path, default=Path("/tmp/CodeRM/data/result/humaneval+"))
    parser.add_argument("--results-root", type=Path, default=Path("/tmp/coderm_extracted/output/humaneval+"))
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()

    results: list[dict] = []
    candidate_rows: list[dict] = []
    for model_index, (model, metadata) in enumerate(MODELS.items()):
        annotation_path = args.annotations_dir / metadata["annotations"]
        result_path = args.results_root / metadata["results_member"]
        if not args.skip_checksums:
            verify(annotation_path, metadata["annotations_sha256"])
            verify(result_path, metadata["results_sha256"])
        task_ids, truth = load_truth(annotation_path)
        scores = load_scores(result_path, task_ids)
        results.append(analyze_model(model_index, model, task_ids, scores, truth))
        for task_index, task_id in enumerate(task_ids):
            for solution_id in range(N_SOLUTIONS):
                candidate_rows.append(
                    {
                        "model": model,
                        "task_id": task_id,
                        "sol_id": solution_id,
                        "verifier_score": f"{scores[task_index, solution_id]:.2f}",
                        "truth": int(truth[task_index, solution_id]),
                    }
                )
    write_outputs(results, candidate_rows)
    print(json.dumps({result["model"]: result["summary"] for result in results}, indent=2))


if __name__ == "__main__":
    main()
