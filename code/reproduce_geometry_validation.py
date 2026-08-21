#!/usr/bin/env python3
"""Reproduce theory checks, empirical analyses, and figures.

The script uses processed summaries from the pinned Monkey Business release
and derived candidate-level scores from the pinned CodeRM release.  Raw-data
reconstruction is implemented in ``analyze_search_spectrum.py`` and
``analyze_coderm_search.py``.  No model API is called here.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import linprog
from scipy.special import eval_chebyu, eval_sh_legendre, roots_legendre
from scipy.stats import norm


DATA = ROOT / "data" / "processed"
FIG = ROOT / "figures"
RESULTS = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
(ROOT / "tmp" / "mplconfig").mkdir(parents=True, exist_ok=True)

SEED = 20260818
TARGET_N = 4096
TAIL_FRACTION = 0.05
MODELS = ["Llama-3-8B-Instruct", "Llama-3-70B-Instruct"]
CODERM_MODELS = ["Llama-3-8B", "Llama-3-70B"]
CODERM_SCALES = np.array([1, 2, 4, 8, 16, 32, 64, 100], dtype=int)
CODERM_BOOTSTRAP = 20_000
CODERM_BOOTSTRAP_SEED = 92_771
PARTIAL_ID_BOOTSTRAP_SEED = 108_203
PARTIAL_ID_GRID_BINS = 1_000
PARTIAL_ID_AUDIT_M = 8
PARTIAL_ID_M_VALUES = np.array([2, 4, 6, 8, 10, 12, 16, 20], dtype=int)
LABEL_ACQUISITION_SEED = 131_071
LABEL_ACQUISITION_REPLICATES = 5_000
LABEL_ACQUISITION_BUDGETS = np.array(
    [25, 50, 100, 250, 500, 1_000, 2_000, 5_000], dtype=int
)
LABEL_ACQUISITION_TARGET_N = 100
LABEL_ACQUISITION_TAIL_FRACTION = 0.05
HELDOUT_TASK_SPLIT_SEED = 260_821
HELDOUT_REPLAY_SEED = 262_147
HELDOUT_DISCOVERY_TAIL_FRACTIONS = np.array([0.05, 0.10, 0.20])
HELDOUT_LABEL_BUDGET = 500
HELDOUT_REPLICATES = 5_000
CERTIFICATE_TARGET_N = 100
CERTIFICATE_ALPHA = 0.05
CERTIFICATE_WIDTHS = np.array([0.50, 0.25, 0.20, 0.10, 0.05])
SHORT_MODEL = {
    "Llama-3-8B-Instruct": "Llama-3-8B",
    "Llama-3-70B-Instruct": "Llama-3-70B",
}
CODERM_SHORT = {
    "Llama-3-8B": "Llama-3-8B",
    "Llama-3-70B": "Llama-3-70B",
}

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B3294"
GRAY = "#5F6368"
LIGHT = "#E8EEF3"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.1,
        "axes.labelsize": 8.2,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.1,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Write both figure formats atomically.

    This prevents an interrupted run from leaving a truncated PDF that LaTeX
    can subsequently mistake for a valid figure.
    """
    pdf_path = FIG / f"{stem}.pdf"
    png_path = FIG / f"{stem}.png"
    pdf_tmp = FIG / f".{stem}.tmp.pdf"
    png_tmp = FIG / f".{stem}.tmp.png"
    fig.savefig(pdf_tmp, format="pdf")
    fig.savefig(png_tmp, format="png", dpi=320)
    os.replace(pdf_tmp, pdf_path)
    os.replace(png_tmp, png_path)
    plt.close(fig)


def exact_blind_width(m: int, n: int) -> float:
    """Exact L1 distance from n*u^(n-1) to polynomials of degree < m."""
    if n <= m:
        return 0.0
    terms = [1.0]
    terms.extend(
        2.0
        * ((-1.0) ** r)
        * math.cos(r * math.pi / (2.0 * (m + 1.0))) ** (2 * n)
        for r in range(1, m + 1)
    )
    result = math.fsum(terms)
    return float(np.clip(result, 0.0, 1.0))


def legendre_witness_gap(m: int, n: int) -> float:
    if n <= m:
        return 0.0
    return float(
        math.exp(
            sum(math.log(n - j) - math.log(n + j) for j in range(1, m + 1))
        )
    )


def regular_legendre_amplitude(m: int, lipschitz: float) -> float:
    """Admissible amplitude for the monotone-Lipschitz Legendre pair."""
    return min(1.0 / (m + 1.0), lipschitz / (m * (m + 1.0)))


def regular_lower_bound(m: int, n: int, lipschitz: float) -> float:
    """Certified gap of the explicit smooth monotone indistinguishable pair."""
    return regular_legendre_amplitude(m, lipschitz) * legendre_witness_gap(m, n)


def gauss_endpoint_radius(m: int) -> float:
    """Endpoint localization radius from a positive polynomial audit kernel.

    If d=floor((m-1)/2), the largest shifted Legendre zero of degree d+1
    minimizes the mean endpoint distance over squared degree-d polynomials.
    """
    degree = (m - 1) // 2 + 1
    nodes, _ = roots_legendre(degree)
    largest_shifted_node = (1.0 + float(nodes[-1])) / 2.0
    return 1.0 - largest_shifted_node


def regular_upper_bound(m: int, n: int, lipschitz: float) -> float:
    """Certified upper bound for monotone L-Lipschitz audit ambiguity."""
    return min(
        1.0,
        exact_blind_width(m, n),
        lipschitz * (gauss_endpoint_radius(m) + 1.0 / (n + 1.0)),
    )


def theta_frontier(tau: float) -> float:
    """Theta-function phase frontier for tau=m^2/N.

    Uses the direct theta_4 series for small tau and its modular transform for
    large tau, avoiding cancellation in the well-validated regime.
    """
    if tau <= 0:
        return 1.0
    if tau <= 1.0:
        total = 1.0
        for r in range(1, 10000):
            term = 2.0 * ((-1.0) ** r) * math.exp(
                -(math.pi**2) * r * r / (4.0 * tau)
            )
            total += term
            if abs(term) < 1e-15:
                break
        return float(np.clip(total, 0.0, 1.0))
    total = 0.0
    for j in range(10000):
        term = math.exp(-tau * (2 * j + 1) ** 2)
        total += term
        if term < 1e-15:
            break
    return float(np.clip(4.0 * math.sqrt(tau / math.pi) * total, 0.0, 1.0))


def chebyshev_truth_worlds(m: int, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sign = np.sign(eval_chebyu(m, 2.0 * u - 1.0))
    sign[sign == 0] = 1.0
    return (1.0 + sign) / 2.0, (1.0 - sign) / 2.0


def verify_exact_frontier() -> list[dict]:
    rows: list[dict] = []
    for m in range(1, 9):
        for n in [m + 1, m + 3, 2 * m + 7, 5 * m + 11]:
            roots = (1.0 + np.cos(np.arange(1, m + 1) * np.pi / (m + 1))) / 2.0
            values = n * roots ** (n - 1)
            if m == 1:
                constant = float(values[0])

                def interpolant(x: float) -> float:
                    return constant

            else:
                # SciPy randomizes the node-product order unless an RNG is
                # supplied.  Pin it so roundoff-level theorem checks are
                # reproducible across repeated runs.
                barycentric = BarycentricInterpolator(
                    roots, values, rng=SEED + 1000 * m + n
                )

                def interpolant(x: float) -> float:
                    return float(barycentric(x))

            numeric_l1 = quad(
                lambda x: abs(n * x ** (n - 1) - interpolant(x)),
                0.0,
                1.0,
                points=np.sort(roots),
                epsabs=2e-12,
                epsrel=2e-12,
                limit=300,
            )[0]
            orthogonality = max(
                abs(
                    quad(
                        lambda x, k=k: np.sign(eval_chebyu(m, 2.0 * x - 1.0))
                        * x**k,
                        0.0,
                        1.0,
                        points=np.sort(roots),
                        epsabs=2e-13,
                        epsrel=2e-13,
                        limit=300,
                    )[0]
                )
                for k in range(m)
            )
            exact = exact_blind_width(m, n)
            rows.append(
                {
                    "m": m,
                    "N": n,
                    "exact_width": f"{exact:.16g}",
                    "numeric_L1": f"{numeric_l1:.16g}",
                    "absolute_error": f"{abs(exact - numeric_l1):.3e}",
                    "max_moment_residual": f"{orthogonality:.3e}",
                }
            )
    write_csv(
        RESULTS / "theorem_checks.csv",
        [
            "m",
            "N",
            "exact_width",
            "numeric_L1",
            "absolute_error",
            "max_moment_residual",
        ],
        rows,
    )
    return rows


def verify_regular_frontier() -> list[dict]:
    """Check the smooth monotone construction and tabulate its sharp-rate bounds."""
    rows: list[dict] = []
    lipschitz = 1.0
    for m in [2, 4, 8, 16, 32, 64]:
        amplitude = regular_legendre_amplitude(m, lipschitz)
        # The shifted derivative has endpoint norm m(m+1).  The two Jordan
        # components have one-sided variation bounded by m+1.
        derivative_bound = amplitude * m * (m + 1)
        grid = np.linspace(0.0, 1.0, 200_001)
        legendre = eval_sh_legendre(m, grid)
        increments = np.diff(legendre)
        positive_height = amplitude * (
            max(float(legendre[0]), 0.0) + np.maximum(increments, 0.0).sum()
        )
        negative_height = amplitude * (
            max(float(-legendre[0]), 0.0) + np.maximum(-increments, 0.0).sum()
        )
        moment_residual = max(
            abs(
                quad(
                    lambda u, degree=degree: eval_sh_legendre(m, u) * u**degree,
                    0.0,
                    1.0,
                    epsabs=2e-13,
                    epsrel=2e-13,
                    limit=300,
                )[0]
            )
            for degree in range(m)
        )
        for tau in [0.25, 0.5, 1.0, 2.0, 4.0]:
            n = max(m + 1, int(round(m * m / tau)))
            rows.append(
                {
                    "m": m,
                    "N": n,
                    "m2_over_N": f"{m*m/n:.12g}",
                    "L": f"{lipschitz:.12g}",
                    "legendre_amplitude": f"{amplitude:.12g}",
                    "explicit_lower_bound": f"{regular_lower_bound(m, n, lipschitz):.12g}",
                    "gauss_upper_bound": f"{regular_upper_bound(m, n, lipschitz):.12g}",
                    "gauss_endpoint_radius": f"{gauss_endpoint_radius(m):.12g}",
                    "constructed_lipschitz": f"{derivative_bound:.12g}",
                    "constructed_f_max": f"{positive_height:.12g}",
                    "constructed_g_max": f"{negative_height:.12g}",
                    "max_audited_moment_residual": f"{moment_residual:.3e}",
                }
            )
    write_csv(
        RESULTS / "regularity_bounds.csv",
        [
            "m",
            "N",
            "m2_over_N",
            "L",
            "legendre_amplitude",
            "explicit_lower_bound",
            "gauss_upper_bound",
            "gauss_endpoint_radius",
            "constructed_lipschitz",
            "constructed_f_max",
            "constructed_g_max",
            "max_audited_moment_residual",
        ],
        rows,
    )
    return rows


def make_frontier_table() -> list[dict]:
    rows: list[dict] = []
    for m in [4, 8, 16, 32, 64, 128]:
        for n in [m + 1, 2 * m, 4 * m, m * m, 4 * m * m, 16 * m * m]:
            rows.append(
                {
                    "m": m,
                    "N": n,
                    "m2_over_N": f"{m*m/n:.12g}",
                    "exact_width": f"{exact_blind_width(m, n):.12g}",
                    "legendre_lower_bound": f"{legendre_witness_gap(m, n):.12g}",
                    "theta_limit": f"{theta_frontier(m*m/n):.12g}",
                }
            )
    write_csv(
        RESULTS / "exact_frontier.csv",
        [
            "m",
            "N",
            "m2_over_N",
            "exact_width",
            "legendre_lower_bound",
            "theta_limit",
        ],
        rows,
    )
    return rows


def make_certificate_planning_table() -> list[dict]:
    """Translate the structural frontier into finite audit-design gates.

    The prefix column is a necessary infinite-task condition: finite sampling
    still has to fit inside the remaining width.  The direct-winner column is
    a separate sufficient design based on a two-sided Hoeffding interval for
    independent bounded task outcomes.
    """
    rows: list[dict] = []
    for width in CERTIFICATE_WIDTHS:
        minimum_m = next(
            m
            for m in range(1, CERTIFICATE_TARGET_N + 1)
            if exact_blind_width(m, CERTIFICATE_TARGET_N) <= width
        )
        direct_tasks = math.ceil(
            2.0 * math.log(2.0 / CERTIFICATE_ALPHA) / (width**2)
        )
        rows.append(
            {
                "target_N": CERTIFICATE_TARGET_N,
                "desired_full_width": f"{width:.12g}",
                "minimum_prefix_m_with_infinite_tasks": minimum_m,
                "structural_width_at_minimum_m": f"{exact_blind_width(minimum_m, CERTIFICATE_TARGET_N):.12g}",
                "direct_winner_tasks_sufficient": direct_tasks,
                "confidence_level": f"{1.0 - CERTIFICATE_ALPHA:.12g}",
            }
        )
    write_csv(
        RESULTS / "certificate_planning.csv",
        list(rows[0].keys()),
        rows,
    )
    return rows


def load_rank_truth() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows = read_csv(DATA / "rank_conditional_truth.csv")
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        mids = np.array([float(row["rank_midpoint"]) for row in subset])
        truth = np.array([float(row["truth_rate"]) for row in subset])
        order = np.argsort(mids)
        output[model] = mids[order], truth[order]
    return output


def piecewise_moment(truth: np.ndarray, power: int) -> float:
    edges = np.linspace(0.0, 1.0, len(truth) + 1)
    return float(np.sum(truth * (edges[1:] ** power - edges[:-1] ** power)))


def design_variance(
    truth: np.ndarray, target_n: int, design_n: int | None = None
) -> tuple[float, float]:
    """Return target and one-observation variance for a beta(design_n,1) audit."""
    target = piecewise_moment(truth, target_n)
    if design_n is None:
        design_n = 1
    second_power = 2 * target_n - design_n
    second = (
        target_n**2
        / (design_n * second_power)
        * piecewise_moment(truth, second_power)
    )
    return target, max(0.0, second - target**2)


def tail_design_variance(
    truth: np.ndarray, target_n: int, tail_fraction: float
) -> tuple[float, float, float]:
    edges = np.linspace(0.0, 1.0, len(truth) + 1)
    cutoff = 1.0 - tail_fraction
    target = piecewise_moment(truth, target_n)
    tail_target = 0.0
    second = 0.0
    for f, lo, hi in zip(truth, edges[:-1], edges[1:]):
        lo2 = max(lo, cutoff)
        if hi <= lo2:
            continue
        tail_target += f * (hi**target_n - lo2**target_n)
        second += (
            tail_fraction
            * target_n**2
            / (2 * target_n - 1)
            * f
            * (hi ** (2 * target_n - 1) - lo2 ** (2 * target_n - 1))
        )
    variance = max(0.0, second - tail_target**2)
    # The omitted contribution is nonnegative and bounded without relying on
    # floating-point subtraction: \int_0^cutoff k_N f <= cutoff^N.
    return target, variance, cutoff**target_n


def make_design_efficiency() -> tuple[list[dict], dict]:
    rank_truth = load_rank_truth()
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for model in MODELS:
        _, truth = rank_truth[model]
        target, deploy_variance = design_variance(truth, TARGET_N, TARGET_N)
        designs = [
            ("uniform candidates", 1),
            ("width-8 winners", 8),
            ("width-64 winners", 64),
            ("deployment winners", TARGET_N),
        ]
        model_summary: dict[str, float] = {"target_truth": target}
        for label, design_n in designs:
            _, variance = design_variance(truth, TARGET_N, design_n)
            ratio = variance / deploy_variance
            rows.append(
                {
                    "model": model,
                    "design": label,
                    "design_n": design_n,
                    "target_N": TARGET_N,
                    "target_truth": f"{target:.12g}",
                    "variance_per_label": f"{variance:.12g}",
                    "labels_per_deployment_label": f"{ratio:.12g}",
                    "absolute_bias": "0",
                }
            )
            model_summary[label] = ratio
        _, tail_variance, tail_bias = tail_design_variance(
            truth, TARGET_N, TAIL_FRACTION
        )
        tail_ratio = tail_variance / deploy_variance
        rows.append(
            {
                "model": model,
                "design": "top-5% rank audit",
                "design_n": "tail",
                "target_N": TARGET_N,
                "target_truth": f"{target:.12g}",
                "variance_per_label": f"{tail_variance:.12g}",
                "labels_per_deployment_label": f"{tail_ratio:.12g}",
                "absolute_bias": f"{tail_bias:.3e}",
            }
        )
        model_summary["top-5% rank audit"] = tail_ratio
        summary[model] = model_summary
    write_csv(
        RESULTS / "design_efficiency.csv",
        [
            "model",
            "design",
            "design_n",
            "target_N",
            "target_truth",
            "variance_per_label",
            "labels_per_deployment_label",
            "absolute_bias",
        ],
        rows,
    )
    return rows, summary


def make_gaussian_selection_table() -> list[dict]:
    alpha = 0.05
    z = norm.ppf(1.0 - alpha / 2.0)
    rows: list[dict] = []
    for targets in [1, 2, 5, 10, 20, 50, 100, 500, 1000, 10000, 100000]:
        coverage = norm.cdf(z) ** targets - norm.cdf(-z) ** targets
        simultaneous = norm.ppf(
            (1.0 + (1.0 - alpha) ** (1.0 / targets)) / 2.0
        )
        rows.append(
            {
                "targets": targets,
                "selected_pointwise_coverage": f"{coverage:.12g}",
                "simultaneous_critical_value": f"{simultaneous:.12g}",
                "width_inflation": f"{simultaneous / z:.12g}",
            }
        )
    write_csv(
        RESULTS / "gaussian_selection.csv",
        [
            "targets",
            "selected_pointwise_coverage",
            "simultaneous_critical_value",
            "width_inflation",
        ],
        rows,
    )
    return rows


def exact_curve_at_scales(
    scores: np.ndarray, truth: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """Exact with-replacement winner reliability with uniform tie breaking."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    group_truth = np.array(
        [sorted_truth[start:end].mean() for start, end in zip(starts, ends)]
    )
    lower = starts / len(scores)
    upper = ends / len(scores)
    return np.array(
        [np.sum(group_truth * (upper**n - lower**n)) for n in scales]
    )


def candidate_winner_probabilities(scores: np.ndarray, width: int) -> np.ndarray:
    """Candidate probabilities under with-replacement best-of-width selection.

    The maximum score wins and ties at the winning score are randomized
    uniformly.  Probabilities are returned in the candidates' original order.
    """
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    sorted_probability = np.empty(len(scores), dtype=float)
    for start, end in zip(starts, ends):
        group_probability = (end / len(scores)) ** width - (
            start / len(scores)
        ) ** width
        sorted_probability[start:end] = group_probability / (end - start)
    probability = np.empty(len(scores), dtype=float)
    probability[order] = sorted_probability
    if not math.isclose(float(probability.sum()), 1.0, abs_tol=5e-13):
        raise RuntimeError("Winner probabilities do not sum to one")
    return probability


def candidate_top_tail_probabilities(
    scores: np.ndarray, tail_fraction: float
) -> np.ndarray:
    """Candidate probabilities for a score-percentile top-tail audit.

    The audit samples a randomized percentile uniformly from the top tail.
    If the tail boundary crosses a score tie, its probability is distributed
    uniformly over every candidate in that tied score group.
    """
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    cutoff = 1.0 - tail_fraction
    sorted_probability = np.zeros(len(scores), dtype=float)
    for start, end in zip(starts, ends):
        overlap = max(
            0.0,
            min(end / len(scores), 1.0) - max(start / len(scores), cutoff),
        )
        sorted_probability[start:end] = (
            overlap / tail_fraction / (end - start)
        )
    probability = np.empty(len(scores), dtype=float)
    probability[order] = sorted_probability
    if not math.isclose(float(probability.sum()), 1.0, abs_tol=5e-13):
        raise RuntimeError("Top-tail probabilities do not sum to one")
    return probability


def make_coderm_label_acquisition() -> list[dict]:
    """Finite masked-label comparison of validation interventions.

    Each audit draw first selects a CodeRM task uniformly, then selects one
    candidate under a score-only intervention.  Candidate truth is treated as
    hidden until that draw.  A Horvitz--Thompson estimate targets the macro
    best-of-100 truth; the top-tail design targets its truncated version and
    carries the analytic omitted-mass bound.  Repeated audits are scored only
    after acquisition against the full-label target.
    """
    source_rows = read_csv(DATA / "coderm_candidate_scores.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in source_rows:
        grouped.setdefault((row["model"], row["task_id"]), []).append(row)

    designs = [
        "uniform candidates",
        "width-8 winners",
        "top-5% rank audit",
        "deployment winners",
    ]
    output_rows: list[dict] = []
    for model_index, model in enumerate(CODERM_MODELS):
        task_keys = sorted(
            [key for key in grouped if key[0] == model],
            key=lambda key: int(key[1].split("/")[-1]),
        )
        target_parts: list[np.ndarray] = []
        truth_parts: list[np.ndarray] = []
        design_parts: dict[str, list[np.ndarray]] = {
            design: [] for design in designs
        }
        for key in task_keys:
            candidates = sorted(
                grouped[key], key=lambda row: int(row["sol_id"])
            )
            scores = np.array(
                [float(row["verifier_score"]) for row in candidates]
            )
            truth = np.array([int(row["truth"]) for row in candidates])
            if len(scores) != 100:
                raise ValueError(f"Expected 100 candidates for {model}, {key[1]}")
            target_probability = candidate_winner_probabilities(
                scores, LABEL_ACQUISITION_TARGET_N
            )
            target_parts.append(target_probability)
            truth_parts.append(truth)
            design_parts["uniform candidates"].append(
                np.full(len(scores), 1.0 / len(scores))
            )
            design_parts["width-8 winners"].append(
                candidate_winner_probabilities(scores, 8)
            )
            design_parts["top-5% rank audit"].append(
                candidate_top_tail_probabilities(
                    scores, LABEL_ACQUISITION_TAIL_FRACTION
                )
            )
            design_parts["deployment winners"].append(target_probability)

        task_count = len(task_keys)
        target_probability = np.concatenate(target_parts) / task_count
        truth = np.concatenate(truth_parts)
        target = float(np.sum(target_probability * truth))
        for design_index, design in enumerate(designs):
            design_probability = np.concatenate(design_parts[design]) / task_count
            supported = design_probability > 0
            observation = np.zeros_like(design_probability)
            observation[supported] = (
                target_probability[supported]
                / design_probability[supported]
                * truth[supported]
            )
            expected_estimand = float(np.sum(design_probability * observation))
            absolute_bias = abs(expected_estimand - target)
            bias_bound = (
                (1.0 - LABEL_ACQUISITION_TAIL_FRACTION)
                ** LABEL_ACQUISITION_TARGET_N
                if design == "top-5% rank audit"
                else 0.0
            )

            rng = np.random.default_rng(
                LABEL_ACQUISITION_SEED + 100 * model_index + design_index
            )
            replicate_sums = np.zeros(LABEL_ACQUISITION_REPLICATES)
            previous_budget = 0
            for budget in LABEL_ACQUISITION_BUDGETS:
                increment = int(budget - previous_budget)
                for start in range(0, LABEL_ACQUISITION_REPLICATES, 250):
                    end = min(start + 250, LABEL_ACQUISITION_REPLICATES)
                    draws = rng.choice(
                        len(design_probability),
                        size=(end - start, increment),
                        replace=True,
                        p=design_probability,
                    )
                    replicate_sums[start:end] += observation[draws].sum(axis=1)
                estimates = replicate_sums / budget
                absolute_error = np.abs(estimates - target)
                output_rows.append(
                    {
                        "model": model,
                        "design": design,
                        "target_N": LABEL_ACQUISITION_TARGET_N,
                        "label_budget": int(budget),
                        "replicates": LABEL_ACQUISITION_REPLICATES,
                        "target_truth": f"{target:.12g}",
                        "expected_estimand": f"{expected_estimand:.12g}",
                        "absolute_bias": f"{absolute_bias:.12g}",
                        "theoretical_bias_bound": f"{bias_bound:.12g}",
                        "median_absolute_error": f"{np.median(absolute_error):.12g}",
                        "q95_absolute_error": f"{np.quantile(absolute_error, 0.95):.12g}",
                        "rmse": f"{np.sqrt(np.mean((estimates - target) ** 2)):.12g}",
                        "maximum_importance_weighted_outcome": f"{observation.max():.12g}",
                    }
                )
                previous_budget = int(budget)

    write_csv(
        RESULTS / "coderm_label_acquisition.csv",
        list(output_rows[0].keys()),
        output_rows,
    )
    return output_rows


def coderm_subset_masked_replay(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
    model: str,
    task_ids: list[str],
    *,
    design: str,
    tail_fraction: float | None,
    seed: int,
) -> dict:
    """Score a fixed audit design on a task subset without using unselected truth.

    The target is the exact macro best-of-100 truth on the supplied tasks.  A
    draw first samples a task uniformly and then a candidate from a score-only
    design.  Full labels are used only after the fixed acquisition rule has
    selected candidates, to score the resulting estimator.
    """
    target_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    design_parts: list[np.ndarray] = []
    for task_id in task_ids:
        candidates = sorted(
            grouped[(model, task_id)], key=lambda row: int(row["sol_id"])
        )
        scores = np.array([float(row["verifier_score"]) for row in candidates])
        truth = np.array([int(row["truth"]) for row in candidates])
        if len(scores) != 100:
            raise ValueError(f"Expected 100 candidates for {model}, {task_id}")
        target_probability = candidate_winner_probabilities(
            scores, LABEL_ACQUISITION_TARGET_N
        )
        if design == "uniform candidates":
            design_probability = np.full(len(scores), 1.0 / len(scores))
        elif design == "top-tail rank audit":
            if tail_fraction is None:
                raise ValueError("A tail fraction is required for top-tail replay")
            design_probability = candidate_top_tail_probabilities(
                scores, tail_fraction
            )
        else:
            raise ValueError(f"Unknown held-out design: {design}")
        target_parts.append(target_probability)
        design_parts.append(design_probability)
        truth_parts.append(truth)

    task_count = len(task_ids)
    target_probability = np.concatenate(target_parts) / task_count
    design_probability = np.concatenate(design_parts) / task_count
    truth = np.concatenate(truth_parts)
    target = float(np.sum(target_probability * truth))
    supported = design_probability > 0
    observation = np.zeros_like(design_probability)
    observation[supported] = (
        target_probability[supported] / design_probability[supported] * truth[supported]
    )
    expected_estimand = float(np.sum(design_probability * observation))

    rng = np.random.default_rng(seed)
    replicate_sums = np.zeros(HELDOUT_REPLICATES)
    for start in range(0, HELDOUT_REPLICATES, 250):
        end = min(start + 250, HELDOUT_REPLICATES)
        draws = rng.choice(
            len(design_probability),
            size=(end - start, HELDOUT_LABEL_BUDGET),
            replace=True,
            p=design_probability,
        )
        replicate_sums[start:end] = observation[draws].sum(axis=1)
    estimates = replicate_sums / HELDOUT_LABEL_BUDGET
    absolute_error = np.abs(estimates - target)
    return {
        "target_truth": target,
        "expected_estimand": expected_estimand,
        "absolute_bias": abs(expected_estimand - target),
        "median_absolute_error": float(np.median(absolute_error)),
        "q95_absolute_error": float(np.quantile(absolute_error, 0.95)),
        "rmse": float(np.sqrt(np.mean((estimates - target) ** 2))),
    }


def make_coderm_heldout_audit() -> dict:
    """Select a tail audit on discovery tasks and evaluate it on held-out tasks.

    A label-blind seeded split partitions the 164 CodeRM tasks into 82 discovery
    and 82 held-out tasks.  Only discovery-task outcomes choose the top-tail
    fraction.  The selected rule is then frozen before the held-out target and
    errors are evaluated.  This remains a retrospective public-data emulation,
    but it prevents within-pool design selection from using held-out outcomes.
    """
    source_rows = read_csv(DATA / "coderm_candidate_scores.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in source_rows:
        grouped.setdefault((row["model"], row["task_id"]), []).append(row)

    task_sets = [
        {task_id for model, task_id in grouped if model == candidate_model}
        for candidate_model in CODERM_MODELS
    ]
    if any(task_set != task_sets[0] for task_set in task_sets[1:]):
        raise RuntimeError("CodeRM models do not share one common task set")
    all_tasks = sorted(task_sets[0], key=lambda task_id: int(task_id.split("/")[-1]))
    rng = np.random.default_rng(HELDOUT_TASK_SPLIT_SEED)
    permutation = rng.permutation(len(all_tasks))
    midpoint = len(all_tasks) // 2
    discovery_tasks = [all_tasks[index] for index in permutation[:midpoint]]
    heldout_tasks = [all_tasks[index] for index in permutation[midpoint:]]

    split_rows = [
        {"task_id": task_id, "split": "discovery"}
        for task_id in sorted(discovery_tasks)
    ] + [
        {"task_id": task_id, "split": "heldout"}
        for task_id in sorted(heldout_tasks)
    ]
    write_csv(
        RESULTS / "coderm_heldout_task_split.csv",
        ["task_id", "split"],
        split_rows,
    )

    rows: list[dict] = []
    discovery_scores: dict[float, float] = {}
    for fraction_index, fraction in enumerate(HELDOUT_DISCOVERY_TAIL_FRACTIONS):
        model_errors = []
        for model_index, model in enumerate(CODERM_MODELS):
            metrics = coderm_subset_masked_replay(
                grouped,
                model,
                discovery_tasks,
                design="top-tail rank audit",
                tail_fraction=float(fraction),
                seed=HELDOUT_REPLAY_SEED + 100 * fraction_index + model_index,
            )
            model_errors.append(metrics["q95_absolute_error"])
            rows.append(
                {
                    "phase": "discovery",
                    "model": model,
                    "design": "candidate top-tail",
                    "tail_fraction": f"{fraction:.12g}",
                    "tasks": len(discovery_tasks),
                    "target_N": LABEL_ACQUISITION_TARGET_N,
                    "label_budget": HELDOUT_LABEL_BUDGET,
                    "replicates": HELDOUT_REPLICATES,
                    "theoretical_omitted_mass_bound": f"{(1.0 - fraction) ** LABEL_ACQUISITION_TARGET_N:.12g}",
                    **{key: f"{value:.12g}" for key, value in metrics.items()},
                }
            )
        discovery_scores[float(fraction)] = float(np.mean(model_errors))

    selected_fraction = min(discovery_scores, key=discovery_scores.get)
    holdout_summary: dict[str, dict] = {}
    for model_index, model in enumerate(CODERM_MODELS):
        holdout_summary[model] = {}
        for design_index, design in enumerate(
            ["uniform candidates", "top-tail rank audit"]
        ):
            metrics = coderm_subset_masked_replay(
                grouped,
                model,
                heldout_tasks,
                design=design,
                tail_fraction=selected_fraction if design == "top-tail rank audit" else None,
                seed=HELDOUT_REPLAY_SEED + 10_000 + 100 * model_index + design_index,
            )
            display_design = (
                "frozen top-tail" if design == "top-tail rank audit" else design
            )
            holdout_summary[model][display_design] = metrics
            rows.append(
                {
                    "phase": "heldout",
                    "model": model,
                    "design": display_design,
                    "tail_fraction": (
                        f"{selected_fraction:.12g}"
                        if design == "top-tail rank audit"
                        else ""
                    ),
                    "tasks": len(heldout_tasks),
                    "target_N": LABEL_ACQUISITION_TARGET_N,
                    "label_budget": HELDOUT_LABEL_BUDGET,
                    "replicates": HELDOUT_REPLICATES,
                    "theoretical_omitted_mass_bound": (
                        f"{(1.0 - selected_fraction) ** LABEL_ACQUISITION_TARGET_N:.12g}"
                        if design == "top-tail rank audit"
                        else "0"
                    ),
                    **{key: f"{value:.12g}" for key, value in metrics.items()},
                }
            )

    write_csv(
        RESULTS / "coderm_heldout_audit.csv",
        list(rows[0].keys()),
        rows,
    )
    return {
        "split_seed": HELDOUT_TASK_SPLIT_SEED,
        "discovery_tasks": len(discovery_tasks),
        "heldout_tasks": len(heldout_tasks),
        "selected_tail_fraction": selected_fraction,
        "selected_omitted_mass_bound": (
            (1.0 - selected_fraction) ** LABEL_ACQUISITION_TARGET_N
        ),
        "discovery_mean_q95_by_tail_fraction": discovery_scores,
        "holdout": holdout_summary,
        "rows": rows,
    }


def tie_averaged_rank_truth(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Represent a discrete score population as a bounded rank--truth law.

    The law is piecewise constant on one equal-width percentile interval per
    candidate.  Every score tie receives its group-average truth value.  Its
    beta(n, 1) moments therefore reproduce the exact uniformly tie-broken
    best-of-n curve for the empirical candidate population.
    """
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    rank_truth = np.empty(len(scores), dtype=float)
    for start, end in zip(starts, ends):
        rank_truth[start:end] = sorted_truth[start:end].mean()
    return rank_truth


def moment_weights(scales: np.ndarray, bins: int) -> np.ndarray:
    """Exact beta(n, 1) masses of equal-width percentile bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    return np.vstack(
        [edges[1:] ** int(n) - edges[:-1] ** int(n) for n in scales]
    )


def solve_bounded_moment_problem(
    audit_matrix: np.ndarray,
    target_weights: np.ndarray,
    *,
    audit_values: np.ndarray | None = None,
    audit_lower: np.ndarray | None = None,
    audit_upper: np.ndarray | None = None,
) -> dict:
    """Extremize a deployment moment over bounded compatible rank--truth laws."""
    if (audit_values is None) == (audit_lower is None or audit_upper is None):
        raise ValueError("Supply either exact audit values or lower/upper audit bands")

    kwargs: dict = {"bounds": (0.0, 1.0), "method": "highs"}
    if audit_values is not None:
        # Raw power moments become nearly collinear as m grows.  QR whitening
        # preserves the exact equality set while preventing solver tolerance
        # in a tiny raw-moment residual from becoming a large target error.
        row_scale = np.linalg.norm(audit_matrix, axis=1)
        if np.any(row_scale == 0):
            raise ValueError("Degenerate audit row")
        scaled = audit_matrix / row_scale[:, None]
        q_basis, r_factor = np.linalg.qr(scaled.T, mode="reduced")
        if np.linalg.matrix_rank(r_factor) < audit_matrix.shape[0]:
            raise ValueError("Audit equalities are numerically rank deficient")
        orthogonal_values = np.linalg.solve(
            r_factor.T, audit_values / row_scale
        )
        kwargs.update(A_eq=q_basis.T, b_eq=orthogonal_values)
    else:
        assert audit_lower is not None and audit_upper is not None
        row_scale = np.linalg.norm(audit_matrix, axis=1)
        if np.any(row_scale == 0):
            raise ValueError("Degenerate audit row")
        scaled = audit_matrix / row_scale[:, None]
        kwargs.update(
            A_ub=np.vstack([scaled, -scaled]),
            b_ub=np.r_[audit_upper / row_scale, -audit_lower / row_scale],
        )

    options = {
        "primal_feasibility_tolerance": 1e-10,
        "dual_feasibility_tolerance": 1e-10,
        "ipm_optimality_tolerance": 1e-10,
    }
    lower_fit = linprog(target_weights, options=options, **kwargs)
    upper_fit = linprog(-target_weights, options=options, **kwargs)
    if not lower_fit.success or not upper_fit.success:
        raise RuntimeError(
            "Partial-identification LP failed: "
            f"lower={lower_fit.message}; upper={upper_fit.message}"
        )

    if audit_values is not None:
        residual = max(
            float(np.max(np.abs(audit_matrix @ lower_fit.x - audit_values))),
            float(np.max(np.abs(audit_matrix @ upper_fit.x - audit_values))),
        )
    else:
        assert audit_lower is not None and audit_upper is not None

        def violation(solution: np.ndarray) -> float:
            observed = audit_matrix @ solution
            return float(
                max(
                    0.0,
                    np.max(observed - audit_upper),
                    np.max(audit_lower - observed),
                )
            )

        residual = max(violation(lower_fit.x), violation(upper_fit.x))

    return {
        "lower": float(lower_fit.fun),
        "upper": float(-upper_fit.fun),
        "lower_world": lower_fit.x,
        "upper_world": upper_fit.x,
        "residual": residual,
    }


def make_coderm_partial_identification() -> dict:
    """Link finite audited search widths to deployment ambiguity in CodeRM.

    For each model, the 164 task-level empirical rank--truth laws are averaged
    to obtain the macro reliability surface.  Linear programs then construct
    bounded piecewise-constant laws that match every plug-in audit from n=1 to
    m while minimizing or maximizing best-of-100 reliability.  A second pair
    of programs propagates a 95% simultaneous task-bootstrap band for those
    audits.  The grid restriction is constructive: every returned extremizer
    is itself a feasible continuum reliability law.
    """
    rows = read_csv(DATA / "coderm_candidate_scores.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["task_id"]), []).append(row)

    all_scales = np.arange(1, 101, dtype=int)
    task_weights = moment_weights(all_scales, 100)
    grid_weights = moment_weights(all_scales, PARTIAL_ID_GRID_BINS)
    target_weights = grid_weights[-1]
    subdivisions = PARTIAL_ID_GRID_BINS // 100
    if subdivisions * 100 != PARTIAL_ID_GRID_BINS:
        raise ValueError("Partial-identification grid must refine the 100 candidate ranks")

    table_rows: list[dict] = []
    world_rows: list[dict] = []
    grid_check_rows: list[dict] = []
    analyses: dict[str, dict] = {}
    for model_index, model in enumerate(CODERM_MODELS):
        task_keys = sorted(
            [key for key in grouped if key[0] == model],
            key=lambda key: int(key[1].split("/")[-1]),
        )
        task_laws = []
        for _, task_id in task_keys:
            candidates = sorted(
                grouped[(model, task_id)], key=lambda row: int(row["sol_id"])
            )
            scores = np.array([float(row["verifier_score"]) for row in candidates])
            truth = np.array([int(row["truth"]) for row in candidates], dtype=float)
            task_laws.append(tie_averaged_rank_truth(scores, truth))
        task_laws = np.vstack(task_laws)
        task_curves = task_laws @ task_weights.T
        observed_curve = task_curves.mean(axis=0)

        rng = np.random.default_rng(PARTIAL_ID_BOOTSTRAP_SEED + model_index)
        bootstrap_counts = rng.multinomial(
            len(task_keys),
            np.full(len(task_keys), 1.0 / len(task_keys)),
            size=CODERM_BOOTSTRAP,
        )
        bootstrap_curves = bootstrap_counts @ task_curves / len(task_keys)
        observed_target_ci = np.quantile(bootstrap_curves[:, -1], [0.025, 0.975])

        plug_in_law = np.repeat(task_laws.mean(axis=0), subdivisions)
        grid_curve = grid_weights @ plug_in_law
        if not np.allclose(grid_curve, observed_curve, atol=5e-13, rtol=0):
            raise RuntimeError(f"Rank-law reconstruction mismatch for {model}")

        model_rows = []
        displayed_exact = None
        for m in PARTIAL_ID_M_VALUES:
            audit_matrix = grid_weights[:m]
            audit_values = grid_curve[:m]
            exact = solve_bounded_moment_problem(
                audit_matrix,
                target_weights,
                audit_values=audit_values,
            )

            bootstrap_audits = bootstrap_curves[:, :m]
            standard_error = bootstrap_audits.std(axis=0, ddof=1)
            max_t = np.max(
                np.abs((bootstrap_audits - observed_curve[:m]) / standard_error),
                axis=1,
            )
            simultaneous_critical = float(np.quantile(max_t, 0.95))
            audit_lower = np.maximum(
                0.0, observed_curve[:m] - simultaneous_critical * standard_error
            )
            audit_upper = np.minimum(
                1.0, observed_curve[:m] + simultaneous_critical * standard_error
            )
            finite = solve_bounded_moment_problem(
                audit_matrix,
                target_weights,
                audit_lower=audit_lower,
                audit_upper=audit_upper,
            )

            row = {
                "model": model,
                "audit_m": int(m),
                "target_N": 100,
                "observed_theta_N": f"{observed_curve[-1]:.12g}",
                "observed_ci_low": f"{observed_target_ci[0]:.12g}",
                "observed_ci_high": f"{observed_target_ci[1]:.12g}",
                "plug_in_lower": f"{exact['lower']:.12g}",
                "plug_in_upper": f"{exact['upper']:.12g}",
                "plug_in_width": f"{exact['upper'] - exact['lower']:.12g}",
                "bootstrap_lower": f"{finite['lower']:.12g}",
                "bootstrap_upper": f"{finite['upper']:.12g}",
                "bootstrap_width": f"{finite['upper'] - finite['lower']:.12g}",
                "simultaneous_critical_value": f"{simultaneous_critical:.12g}",
                "max_exact_audit_residual": f"{exact['residual']:.3e}",
                "max_bootstrap_band_violation": f"{finite['residual']:.3e}",
                "grid_bins": PARTIAL_ID_GRID_BINS,
                "bootstrap_replicates": CODERM_BOOTSTRAP,
            }
            table_rows.append(row)
            model_rows.append(row)
            if m == PARTIAL_ID_AUDIT_M:
                displayed_exact = exact

        if displayed_exact is None:
            raise RuntimeError("Displayed audit width missing from partial-ID grid")

        for check_bins in [200, 500, PARTIAL_ID_GRID_BINS]:
            if check_bins % 100:
                raise ValueError("Grid check must refine the 100 empirical rank intervals")
            check_weights = moment_weights(all_scales, check_bins)
            check_law = np.repeat(task_laws.mean(axis=0), check_bins // 100)
            check_fit = solve_bounded_moment_problem(
                check_weights[:PARTIAL_ID_AUDIT_M],
                check_weights[-1],
                audit_values=check_weights[:PARTIAL_ID_AUDIT_M] @ check_law,
            )
            grid_check_rows.append(
                {
                    "model": model,
                    "audit_m": PARTIAL_ID_AUDIT_M,
                    "target_N": 100,
                    "grid_bins": check_bins,
                    "lower": f"{check_fit['lower']:.12g}",
                    "upper": f"{check_fit['upper']:.12g}",
                    "width": f"{check_fit['upper'] - check_fit['lower']:.12g}",
                    "max_audit_residual": f"{check_fit['residual']:.3e}",
                }
            )
        lower_world = displayed_exact["lower_world"]
        upper_world = displayed_exact["upper_world"]
        bin_edges = np.linspace(0.0, 1.0, PARTIAL_ID_GRID_BINS + 1)
        bin_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        for law_name, law in [
            ("empirical_plugin", plug_in_law),
            ("lower_extremizer", lower_world),
            ("upper_extremizer", upper_world),
        ]:
            for index, (midpoint, value) in enumerate(zip(bin_midpoints, law)):
                world_rows.append(
                    {
                        "model": model,
                        "audit_m": PARTIAL_ID_AUDIT_M,
                        "target_N": 100,
                        "law": law_name,
                        "bin_index": index,
                        "rank_midpoint": f"{midpoint:.12g}",
                        "truth_rate": f"{value:.12g}",
                    }
                )

        analyses[model] = {
            "observed_curve": grid_curve,
            "lower_curve": grid_weights @ lower_world,
            "upper_curve": grid_weights @ upper_world,
            "plug_in_law": plug_in_law,
            "lower_world": lower_world,
            "upper_world": upper_world,
            "target_ci": observed_target_ci,
            "rows": model_rows,
        }

    write_csv(
        RESULTS / "coderm_partial_identification.csv",
        list(table_rows[0].keys()),
        table_rows,
    )
    write_csv(
        RESULTS / "coderm_partial_id_worlds.csv",
        [
            "model",
            "audit_m",
            "target_N",
            "law",
            "bin_index",
            "rank_midpoint",
            "truth_rate",
        ],
        world_rows,
    )
    write_csv(
        RESULTS / "coderm_partial_id_grid_check.csv",
        list(grid_check_rows[0].keys()),
        grid_check_rows,
    )
    return {"models": analyses, "rows": table_rows}


def load_coderm_analysis() -> dict:
    """Recompute the CodeRM replication from derived candidate-level numbers."""
    rows = read_csv(DATA / "coderm_candidate_scores.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["task_id"]), []).append(row)

    spectra: list[dict] = []
    effects: list[dict] = []
    summaries: dict[str, dict] = {}
    for model_index, model in enumerate(CODERM_MODELS):
        task_keys = sorted(
            [key for key in grouped if key[0] == model],
            key=lambda key: int(key[1].split("/")[-1]),
        )
        curves: list[np.ndarray] = []
        model_effects: list[dict] = []
        aucs: list[float] = []
        for _, task_id in task_keys:
            candidates = sorted(grouped[(model, task_id)], key=lambda row: int(row["sol_id"]))
            if len(candidates) != 100:
                raise ValueError(f"Expected 100 CodeRM candidates for {model}, {task_id}")
            scores = np.array([float(row["verifier_score"]) for row in candidates])
            truth = np.array([int(row["truth"]) for row in candidates])
            curve = exact_curve_at_scales(scores, truth, CODERM_SCALES)
            curves.append(curve)
            positive = scores[truth == 1]
            negative = scores[truth == 0]
            if len(positive) and len(negative):
                comparisons = positive[:, None] - negative[None, :]
                auc = float(
                    (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
                    / comparisons.size
                )
            else:
                auc = float("nan")
            aucs.append(auc)
            model_effects.append(
                {
                    "model": model,
                    "task_id": task_id,
                    "delta": float(curve[-1] - curve[0]),
                    "within_task_auc": auc,
                }
            )

        matrix = np.vstack(curves)
        rng = np.random.default_rng(CODERM_BOOTSTRAP_SEED + model_index)
        bootstrap_indices = rng.integers(
            0, len(task_keys), size=(CODERM_BOOTSTRAP, len(task_keys))
        )
        bootstrap = matrix[bootstrap_indices].mean(axis=1)
        low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        auc_array = np.asarray(aucs)
        bootstrap_auc = np.nanmean(auc_array[bootstrap_indices], axis=1)
        auc_low, auc_high = np.quantile(bootstrap_auc, [0.025, 0.975])
        for n, mean, lo, hi in zip(CODERM_SCALES, matrix.mean(axis=0), low, high):
            spectra.append(
                {
                    "model": model,
                    "n": int(n),
                    "mean": float(mean),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                }
            )
        effects.extend(model_effects)
        deltas = matrix[:, -1] - matrix[:, 0]
        summaries[model] = {
            "tasks": len(task_keys),
            "candidates": int(matrix.shape[0] * 100),
            "truth_n1": float(matrix[:, 0].mean()),
            "truth_n100": float(matrix[:, -1].mean()),
            "macro_auc": float(np.nanmean(auc_array)),
            "macro_auc_ci95": [float(auc_low), float(auc_high)],
            "tasks_harmed_gt_1pp": int(np.sum(deltas < -0.01)),
            "worst_delta": float(deltas.min()),
        }

    with (DATA / "coderm_summary.json").open("r", encoding="utf-8") as stream:
        pinned = json.load(stream)["models"]
    for model in CODERM_MODELS:
        for key in ["truth_n1", "truth_n100", "macro_auc", "worst_delta"]:
            if not math.isclose(summaries[model][key], pinned[model][key], abs_tol=5e-12):
                raise RuntimeError(f"CodeRM lightweight reconstruction mismatch: {model} {key}")

    output = {"models": summaries, "search_scales": CODERM_SCALES.tolist()}
    with (RESULTS / "coderm_recomputed_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    return {"spectrum": spectra, "effects": effects, "summary": summaries}


def figure_general_geometry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), constrained_layout=True)

    ax = axes[0]
    ax.axhline(0.0, color=GRAY, lw=2.0)
    points = np.array([[-0.82, 0.10], [-0.35, -0.17], [0.08, 0.25], [0.50, -0.37], [0.78, 0.82]])
    for idx, (x, y) in enumerate(points):
        color = ORANGE if idx == len(points) - 1 else BLUE
        size = 48 if idx == len(points) - 1 else 31
        ax.plot([x, x], [0, y], color=color, lw=1.0, ls="--", alpha=0.9)
        ax.scatter([x], [y], s=size, color=color, zorder=3)
        ax.scatter([x], [0], s=16, facecolor="white", edgecolor=color, zorder=3)
    ax.annotate(
        "adaptively selected\ntarget",
        xy=(0.78, 0.82),
        xytext=(0.05, 1.02),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.0},
        color=ORANGE,
        ha="left",
        va="center",
    )
    ax.annotate(
        r"blind residual $b(k,V)$",
        xy=(0.78, 0.42),
        xytext=(0.02, 0.58),
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.9},
        color=GRAY,
    )
    ax.text(-0.94, -0.09, r"validated span $S_V$", color=GRAY, va="top")
    ax.text(-0.95, 0.93, "deployment kernels", color=BLUE, va="top")
    ax.set(xlim=(-1.0, 1.0), ylim=(-0.62, 1.12), title="A  Validation is target-relative")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_visible(False)

    ax = axes[1]
    alpha = 0.05
    z = norm.ppf(1.0 - alpha / 2.0)
    targets = np.unique(np.geomspace(1, 100000, 500).astype(int))
    coverage = norm.cdf(z) ** targets - norm.cdf(-z) ** targets
    simultaneous = norm.ppf(
        (1.0 + (1.0 - alpha) ** (1.0 / targets)) / 2.0
    )
    ax.plot(targets, coverage, color=ORANGE, lw=2.0, label="Coverage after selecting max")
    ax.axhline(0.95, color=GRAY, ls="--", lw=1.0, label="Fixed-target coverage")
    ax.set(
        xscale="log",
        xlim=(1, 100000),
        ylim=(-0.02, 1.02),
        xlabel="Number of selectable targets, $M$",
        ylabel="Coverage of nominal 95% interval",
        title="B  Pointwise validity is not selected validity",
    )
    ax2 = ax.twinx()
    ax2.plot(targets, simultaneous / z, color=PURPLE, lw=1.7, label="Simultaneous width penalty")
    ax2.set(ylabel="Required half-width / pointwise half-width", ylim=(0.9, 2.8))
    lines = ax.get_lines()[:2] + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="center left")
    save_figure(fig, "fig1_geometry_adaptive_targets")


def figure_exact_frontier() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0), constrained_layout=True)
    m = 6

    ax = axes[0, 0]
    u = np.linspace(0.0, 1.0, 2001)
    f_plus, f_minus = chebyshev_truth_worlds(m, u)
    ax.step(u, f_plus, where="mid", color=BLUE, lw=1.5, label=r"$f_+$")
    ax.step(u, f_minus, where="mid", color=ORANGE, lw=1.5, label=r"$f_-$")
    roots = np.sort((1.0 + np.cos(np.arange(1, m + 1) * np.pi / (m + 1))) / 2.0)
    for root in roots:
        ax.axvline(root, color=LIGHT, lw=0.7, zorder=-2)
    ax.set(
        xlim=(0, 1),
        ylim=(-0.03, 1.03),
        xlabel="Score percentile $u$",
        ylabel="Truth probability",
        title="A  Exact blind worlds",
    )
    ax.legend(frameon=False, loc="upper center", ncol=2)

    ax = axes[0, 1]
    n_values = np.unique(np.r_[np.arange(1, m + 2), np.geomspace(m + 1, 100000, 450).astype(int)])
    widths = np.array([exact_blind_width(m, int(n)) for n in n_values])
    ax.axvspan(1, m, color=LIGHT, zorder=-3)
    ax.plot(n_values, 0.5 * (1 + widths), color=BLUE, lw=2.0, label="World +")
    ax.plot(n_values, 0.5 * (1 - widths), color=ORANGE, lw=2.0, label="World -")
    ax.axhline(0.5, color=GRAY, ls="--", lw=1.0)
    ax.set(
        xscale="log",
        xlim=(1, 100000),
        ylim=(-0.02, 1.02),
        xlabel="Candidates searched, $n$",
        ylabel="Selected truth rate",
        title="B  Exact agreement, maximal separation",
    )
    ax.text(2.5, 0.91, "audited", color=GRAY, ha="center")
    ax.legend(frameon=False, loc="center left")

    ax = axes[1, 0]
    tau_grid = np.geomspace(0.025, 12.0, 350)
    for mm, color in zip([16, 32, 64, 128], [PURPLE, GREEN, ORANGE, BLUE]):
        exact_x = []
        exact_y = []
        for tau in tau_grid:
            nn = max(mm + 1, int(round(mm * mm / tau)))
            actual_tau = mm * mm / nn
            if actual_tau <= 12.2:
                exact_x.append(actual_tau)
                exact_y.append(exact_blind_width(mm, nn))
        ax.plot(exact_x, exact_y, color=color, lw=1.25, alpha=0.82, label=f"$m={mm}$")
    limit = np.array([theta_frontier(tau) for tau in tau_grid])
    ax.plot(tau_grid, limit, color="black", lw=1.8, ls="--", label=r"exact phase limit")
    ax.plot(tau_grid, np.exp(-tau_grid), color=GRAY, lw=1.0, ls=":", label=r"Legendre witness $e^{-m^2/N}$")
    ax.set(
        xscale="log",
        xlim=(0.025, 12),
        ylim=(-0.02, 1.02),
        xlabel=r"Audit ratio $m^2/N$",
        ylabel="Worst-case identified width",
        title="C  The search-validation frontier",
    )
    ax.legend(frameon=False, loc="lower left", ncol=1, fontsize=6.2)

    ax = axes[1, 1]
    regular_m = 64
    lipschitz = 1.0
    regular_tau = np.geomspace(0.08, 6.0, 250)
    finite_tau = []
    lower = []
    upper = []
    for tau in regular_tau:
        nn = max(regular_m + 1, int(round(regular_m * regular_m / tau)))
        finite_tau.append(regular_m * regular_m / nn)
        lower.append(
            regular_m**2 * regular_lower_bound(regular_m, nn, lipschitz) / lipschitz
        )
        upper.append(
            regular_m**2 * regular_upper_bound(regular_m, nn, lipschitz) / lipschitz
        )
    finite_tau = np.asarray(finite_tau)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    order = np.argsort(finite_tau)
    ax.fill_between(
        finite_tau[order],
        lower[order],
        upper[order],
        color=LIGHT,
        alpha=0.9,
        label="certified region ($m=64$)",
    )
    ax.plot(
        finite_tau[order],
        lower[order],
        color=ORANGE,
        lw=1.8,
        label="explicit smooth pair",
    )
    ax.plot(
        finite_tau[order],
        upper[order],
        color=BLUE,
        lw=1.8,
        label="positive-kernel upper bound",
    )
    ax.plot(
        regular_tau,
        np.exp(-regular_tau),
        color="black",
        lw=1.1,
        ls="--",
        label=r"lower limit $e^{-m^2/N}$",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(0.08, 6.0),
        ylim=(1e-3, 20),
        xlabel=r"Audit ratio $m^2/N$",
        ylabel=r"Normalized ambiguity $m^2\Delta/L$",
        title="D  Monotone smooth worlds remain blind",
    )
    ax.legend(frameon=False, loc="lower left", fontsize=6.2)
    save_figure(fig, "fig2_exact_best_of_n_frontier")


def figure_empirical(label_rows: list[dict], coderm: dict) -> None:
    spectrum = read_csv(DATA / "search_spectrum.csv")
    effects = read_csv(DATA / "problem_effects.csv")
    splits = read_csv(DATA / "split_robustness.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.0), constrained_layout=True)

    ax = axes[0, 0]
    for model, color in zip(MODELS, [BLUE, ORANGE]):
        subset = [row for row in spectrum if row["model"] == model]
        n = np.array([int(row["n"]) for row in subset])
        mean = np.array([float(row["mean"]) for row in subset])
        low = np.array([float(row["ci_low"]) for row in subset])
        high = np.array([float(row["ci_high"]) for row in subset])
        ax.fill_between(n, low, high, color=color, alpha=0.12)
        ax.plot(n, mean, color=color, lw=2.0, marker="o", ms=2.5, label=SHORT_MODEL[model])
    ax.set(
        xscale="log",
        xlim=(1, 4096),
        ylim=(0.65, 1.005),
        xlabel="Candidates searched, $n$",
        ylabel="Mean winner truth",
        title="A  Math: 127 problems",
    )
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    for model, color in zip(CODERM_MODELS, [BLUE, ORANGE]):
        subset = [row for row in coderm["spectrum"] if row["model"] == model]
        n = np.array([int(row["n"]) for row in subset])
        mean = np.array([float(row["mean"]) for row in subset])
        low = np.array([float(row["ci_low"]) for row in subset])
        high = np.array([float(row["ci_high"]) for row in subset])
        ax.fill_between(n, low, high, color=color, alpha=0.12)
        ax.plot(n, mean, color=color, lw=2.0, marker="o", ms=2.5, label=CODERM_SHORT[model])
    ax.set(
        xscale="log",
        xlim=(1, 100),
        ylim=(0.45, 0.86),
        xlabel="Programs searched, $n$",
        ylabel="Mean winner truth",
        title="B  Code: 164 tasks",
    )
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 2]
    for model, color in zip(MODELS, [BLUE, ORANGE]):
        delta = np.sort(
            np.array(
                [float(row["delta"]) for row in effects if row["model"] == model]
            )
        )
        x = np.linspace(0.0, 1.0, len(delta))
        ax.plot(x, delta, color=color, lw=1.8, label=SHORT_MODEL[model])
    ax.axhline(0.0, color=GRAY, ls="--", lw=1.0)
    ax.set(
        xlim=(0, 1),
        ylim=(-0.52, 0.92),
        xlabel="Problem quantile",
        ylabel=r"$\Delta$ truth, $1\to4096$",
        title="C  Math reversals",
    )
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 0]
    for model, color in zip(CODERM_MODELS, [BLUE, ORANGE]):
        delta = np.sort(
            np.array(
                [float(row["delta"]) for row in coderm["effects"] if row["model"] == model]
            )
        )
        x = np.linspace(0.0, 1.0, len(delta))
        ax.plot(x, delta, color=color, lw=1.8, label=CODERM_SHORT[model])
    ax.axhline(0.0, color=GRAY, ls="--", lw=1.0)
    ax.set(
        xlim=(0, 1),
        ylim=(-0.78, 0.98),
        xlabel="Task quantile",
        ylabel=r"$\Delta$ truth, $1\to100$",
        title="D  Code reversals",
    )
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 1]
    width = 0.35
    split_ids = np.arange(10)
    for offset, model, color in zip([-width / 2, width / 2], MODELS, [BLUE, ORANGE]):
        subset = sorted(
            [row for row in splits if row["model"] == model], key=lambda row: int(row["split"])
        )
        harmed = np.array([int(row["problems_harmed_gt_1pp"]) for row in subset])
        ax.bar(split_ids + offset, harmed, width=width, color=color, alpha=0.88, label=SHORT_MODEL[model])
    ax.set(
        xticks=split_ids,
        xlabel="Split",
        ylabel="Problems harmed $>1$ pp",
        title="E  Split persistence",
        ylim=(0, 17),
    )
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)

    ax = axes[1, 2]
    design_order = [
        "uniform candidates",
        "width-8 winners",
        "top-5% rank audit",
        "deployment winners",
    ]
    labels = ["Uniform", "Bo8", "Top 5%", "Bo100"]
    x = np.arange(len(design_order))
    for offset, model, color, hatch in zip(
        [-width / 2, width / 2], CODERM_MODELS, [BLUE, ORANGE], ["", "///"]
    ):
        errors = []
        for design in design_order:
            row = next(
                row
                for row in label_rows
                if row["model"] == model
                and row["design"] == design
                and int(row["label_budget"]) == 500
            )
            errors.append(float(row["q95_absolute_error"]))
        ax.bar(
            x + offset,
            errors,
            width=width,
            color=color,
            alpha=0.86,
            hatch=hatch,
            label=CODERM_SHORT[model],
        )
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylim=(0.0, 0.21),
        ylabel="95th-pct. absolute error",
        title="F  500 masked labels",
    )
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    for panel in axes.flat:
        panel.tick_params(labelsize=6.7)
    save_figure(fig, "fig3_empirical_validation_design")


def figure_coderm_partial_identification(partial_id: dict) -> None:
    """Show empirical rank--truth laws that audits through m=8 cannot separate."""
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.75), constrained_layout=True)
    scales = np.arange(1, 101)
    rank = (np.arange(PARTIAL_ID_GRID_BINS) + 0.5) / PARTIAL_ID_GRID_BINS

    for column, model in enumerate(CODERM_MODELS):
        analysis = partial_id["models"][model]
        row = next(
            row
            for row in analysis["rows"]
            if int(row["audit_m"]) == PARTIAL_ID_AUDIT_M
        )

        ax = axes[0, column]
        ax.plot(
            rank,
            analysis["lower_world"],
            color=BLUE,
            lw=0.9,
            alpha=0.9,
            drawstyle="steps-mid",
            label="Lowest compatible target",
        )
        ax.plot(
            rank,
            analysis["upper_world"],
            color=ORANGE,
            lw=0.9,
            alpha=0.9,
            drawstyle="steps-mid",
            label="Highest compatible target",
        )
        ax.plot(
            rank,
            analysis["plug_in_law"],
            color="black",
            lw=1.5,
            label="Empirical plug-in law",
        )
        ax.set(
            xlim=(0.0, 1.0),
            ylim=(-0.03, 1.03),
            xlabel="Verifier-score percentile, $u$",
            ylabel="Truth probability" if column == 0 else None,
            title=("A" if column == 0 else "B")
            + f"  {CODERM_SHORT[model]}: compatible rank--truth laws",
        )
        if column == 0:
            ax.legend(frameon=False, loc="lower left", fontsize=6.4)

        ax = axes[1, column]
        ax.axvspan(1, PARTIAL_ID_AUDIT_M, color=LIGHT, alpha=0.9, zorder=-3)
        ax.plot(
            scales,
            analysis["lower_curve"],
            color=BLUE,
            lw=1.8,
            label="Compatible lower curve",
        )
        ax.plot(
            scales,
            analysis["upper_curve"],
            color=ORANGE,
            lw=1.8,
            label="Compatible upper curve",
        )
        ax.plot(
            scales,
            analysis["observed_curve"],
            color="black",
            lw=1.5,
            label="Observed curve",
        )
        audit_scales = np.arange(1, PARTIAL_ID_AUDIT_M + 1)
        ax.scatter(
            audit_scales,
            analysis["observed_curve"][:PARTIAL_ID_AUDIT_M],
            color="black",
            s=10,
            zorder=4,
        )
        ax.axvline(100, color=GRAY, lw=0.8, ls=":")
        ax.set(
            xscale="log",
            xlim=(1, 100),
            ylim=(-0.03, 1.03),
            xticks=[1, 2, 4, 8, 16, 32, 64, 100],
            xticklabels=["1", "2", "4", "8", "16", "32", "64", "100"],
            xlabel="Programs searched, $n$",
            ylabel="Winner truth" if column == 0 else None,
            title=("C" if column == 0 else "D")
            + f"  Agreement through $m={PARTIAL_ID_AUDIT_M}$, divergence at $N=100$",
        )
        ax.text(
            1.25,
            0.05,
            "audited",
            color=GRAY,
            fontsize=6.7,
            ha="left",
        )
        ax.text(
            92,
            float(row["plug_in_upper"]) - 0.025,
            f"{float(row['plug_in_upper']):.3f}",
            color=ORANGE,
            fontsize=6.5,
            ha="right",
            va="top",
        )
        ax.text(
            92,
            float(row["plug_in_lower"]) + 0.025,
            f"{float(row['plug_in_lower']):.3f}",
            color=BLUE,
            fontsize=6.5,
            ha="right",
            va="bottom",
        )
        if column == 0:
            ax.legend(frameon=False, loc="center left", fontsize=6.4)

    save_figure(fig, "fig4_empirical_partial_identification")


def write_extended_summary(
    theorem_rows: list[dict],
    regular_rows: list[dict],
    design_summary: dict,
    gaussian_rows: list[dict],
    coderm: dict,
    partial_id: dict,
    label_rows: list[dict],
    heldout_audit: dict,
    certificate_rows: list[dict],
) -> None:
    with (DATA / "summary.json").open("r", encoding="utf-8") as stream:
        source_summary = json.load(stream)
    max_frontier_error = max(float(row["absolute_error"]) for row in theorem_rows)
    max_moment_error = max(float(row["max_moment_residual"]) for row in theorem_rows)
    output = {
        "analysis_seed": SEED,
        "target_search_width": TARGET_N,
        "source_empirical_summary": source_summary,
        "exact_frontier_verification": {
            "cases": len(theorem_rows),
            "maximum_absolute_L1_error": max_frontier_error,
            "maximum_moment_residual": max_moment_error,
        },
        "shape_restricted_verification": {
            "cases": len(regular_rows),
            "maximum_audited_moment_residual": max(
                float(row["max_audited_moment_residual"]) for row in regular_rows
            ),
            "maximum_constructed_lipschitz": max(
                float(row["constructed_lipschitz"]) for row in regular_rows
            ),
            "maximum_constructed_world_value": max(
                max(float(row["constructed_f_max"]), float(row["constructed_g_max"]))
                for row in regular_rows
            ),
        },
        "intervention_efficiency": design_summary,
        "coderm_replication": coderm["summary"],
        "coderm_partial_identification_m8": {
            model: {
                key: next(
                    row[key]
                    for row in partial_id["rows"]
                    if row["model"] == model
                    and int(row["audit_m"]) == PARTIAL_ID_AUDIT_M
                )
                for key in [
                    "observed_theta_N",
                    "observed_ci_low",
                    "observed_ci_high",
                    "plug_in_lower",
                    "plug_in_upper",
                    "bootstrap_lower",
                    "bootstrap_upper",
                    "max_exact_audit_residual",
                ]
            }
            for model in CODERM_MODELS
        },
        "coderm_masked_label_acquisition": {
            "replicates": LABEL_ACQUISITION_REPLICATES,
            "budget_500_q95_absolute_error": {
                model: {
                    design: next(
                        row["q95_absolute_error"]
                        for row in label_rows
                        if row["model"] == model
                        and row["design"] == design
                        and int(row["label_budget"]) == 500
                    )
                    for design in [
                        "uniform candidates",
                        "width-8 winners",
                        "top-5% rank audit",
                        "deployment winners",
                    ]
                }
                for model in CODERM_MODELS
            },
        },
        "coderm_heldout_audit": {
            "split_seed": heldout_audit["split_seed"],
            "discovery_tasks": heldout_audit["discovery_tasks"],
            "heldout_tasks": heldout_audit["heldout_tasks"],
            "selected_tail_fraction": heldout_audit["selected_tail_fraction"],
            "selected_omitted_mass_bound": heldout_audit[
                "selected_omitted_mass_bound"
            ],
            "discovery_mean_q95_by_tail_fraction": heldout_audit[
                "discovery_mean_q95_by_tail_fraction"
            ],
            "holdout_budget_500": heldout_audit["holdout"],
        },
        "finite_certificate_planning": certificate_rows,
        "gaussian_selection": gaussian_rows,
    }
    with (RESULTS / "extended_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


def main() -> None:
    theorem_rows = verify_exact_frontier()
    regular_rows = verify_regular_frontier()
    make_frontier_table()
    certificate_rows = make_certificate_planning_table()
    design_rows, design_summary = make_design_efficiency()
    gaussian_rows = make_gaussian_selection_table()
    coderm = load_coderm_analysis()
    partial_id = make_coderm_partial_identification()
    label_rows = make_coderm_label_acquisition()
    heldout_audit = make_coderm_heldout_audit()
    figure_general_geometry()
    figure_exact_frontier()
    figure_empirical(label_rows, coderm)
    figure_coderm_partial_identification(partial_id)
    write_extended_summary(
        theorem_rows,
        regular_rows,
        design_summary,
        gaussian_rows,
        coderm,
        partial_id,
        label_rows,
        heldout_audit,
        certificate_rows,
    )
    print(json.dumps(design_summary, indent=2))


if __name__ == "__main__":
    main()
