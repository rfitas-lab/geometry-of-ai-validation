#!/usr/bin/env python3
"""Check that the empirical conclusions are not specific to one split seed."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score

from analyze_search_spectrum import (
    FILES,
    HIGH_SCALE,
    N_DEPLOYMENT,
    N_REFERENCE,
    RAW,
    PROCESSED,
    SPLIT_SEED,
    exact_winner_curve,
    normalize_answer,
    sha256,
)


N_SPLITS = 10


def main() -> None:
    rows: list[dict] = []
    for model_index, (model, (filename, expected_sha)) in enumerate(FILES.items()):
        path = RAW / filename
        if not path.exists() or sha256(path) != expected_sha:
            raise FileNotFoundError(f"Missing or invalid pinned input: {path}")
        print(f"Loading and normalizing {filename} ...")
        with path.open(encoding="utf-8") as stream:
            records = json.load(stream)
        normalized = [
            [normalize_answer(sample) for sample in record["samples"]]
            for record in records
        ]

        for split in range(N_SPLITS):
            deltas: list[float] = []
            aucs: list[float] = []
            for problem_index, (record, answers) in enumerate(zip(records, normalized)):
                labels = np.asarray(record["is_corrects"], dtype=float)
                rng = np.random.default_rng(
                    SPLIT_SEED
                    + 1_000_000 * split
                    + 10_000 * model_index
                    + problem_index
                )
                indices = rng.permutation(len(answers))
                reference_indices = indices[:N_REFERENCE]
                deployment_indices = indices[N_REFERENCE:]
                counts = Counter(
                    answers[i] for i in reference_indices if answers[i] is not None
                )
                scores = np.asarray(
                    [
                        counts.get(answers[i], 0) / N_REFERENCE
                        if answers[i] is not None
                        else -1.0 / N_REFERENCE
                        for i in deployment_indices
                    ],
                    dtype=float,
                )
                truth = labels[deployment_indices]
                if len(truth) != N_DEPLOYMENT:
                    raise ValueError("Unexpected deployment population size")
                curve, _ = exact_winner_curve(scores, truth)
                deltas.append(float(curve[-1] - curve[0]))
                if len(np.unique(truth)) == 2:
                    aucs.append(float(roc_auc_score(truth, scores)))

            delta_array = np.asarray(deltas)
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "mean_delta": float(delta_array.mean()),
                    "problems_harmed_gt_1pp": int(np.sum(delta_array < -0.01)),
                    "problems_harmed_gt_5pp": int(np.sum(delta_array < -0.05)),
                    "worst_delta": float(delta_array.min()),
                    "macro_auc": float(np.mean(aucs)),
                }
            )
            print(model, split, rows[-1])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / "split_robustness.csv"
    fields = list(rows[0])
    with path.open("w", encoding="utf-8") as stream:
        stream.write(",".join(fields) + "\n")
        for row in rows:
            stream.write(",".join(str(row[field]) for field in fields) + "\n")

    print("\nRanges across splits:")
    for model in FILES:
        group = [row for row in rows if row["model"] == model]
        print(model)
        for field in fields[2:]:
            values = np.asarray([row[field] for row in group], dtype=float)
            print(f"  {field}: {values.min():.6g} to {values.max():.6g}")


if __name__ == "__main__":
    main()
