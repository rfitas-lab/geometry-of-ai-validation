#!/usr/bin/env python3
"""Numerically evaluate the exact monotone-Lipschitz capped-tail dual.

The continuum identity is

    Delta = inf_a {H_L(z_a) + H_L(-z_a)}.

For numerical checks, q is restricted to be constant on equal-width bins.  The
resulting primal and its bin-averaged capped-knapsack dual are finite linear
programs with identical optima.  A returned dual coefficient vector is then
evaluated in the continuum support formula

    H_L(z) = inf_{lambda >= 0}
             lambda + (z(0)-lambda)_+
             + L integral (z(t)-lambda)_+ dt.

The bin primal is a feasible lower bound; the continuum-evaluated dual is an
upper bound up to the reported numerical root/integration tolerance.  Nested
bin counts document convergence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy import sparse
from scipy.optimize import brentq, linprog, minimize_scalar
from scipy.special import roots_legendre


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def bin_integrals(m: int, n_target: int, bins: int):
    h = 1.0 / bins
    edges = np.arange(bins + 1, dtype=float) / bins
    audit = np.array(
        [
            h
            - (edges[1:] ** (n + 1) - edges[:-1] ** (n + 1))
            / (n + 1)
            for n in range(1, m + 1)
        ]
    )
    target = h - (
        edges[1:] ** (n_target + 1) - edges[:-1] ** (n_target + 1)
    ) / (n_target + 1)
    return h, audit, target


def solve_discrete(m: int, n_target: int, lipschitz: float, bins: int):
    """Solve the equal-bin primal and its exact finite LP dual."""
    h, audit_int, target_int = bin_integrals(m, n_target, bins)
    audit_avg = audit_int / h
    target_avg = target_int / h

    # Dual variables:
    # a[0:m], lambda_plus, lambda_minus, atom_slack_plus,
    # atom_slack_minus, bin_slack_plus[0:K], bin_slack_minus[0:K].
    i_lp = m
    i_lm = m + 1
    i_ap = m + 2
    i_am = m + 3
    i_rp = m + 4
    i_rm = i_rp + bins
    n_var = i_rm + bins

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row = 0

    def add_support_constraint(sign: int, *, atom: bool, j: int = 0):
        nonlocal row
        values = np.ones(m) if atom else audit_avg[:, j]
        for n, value in enumerate(values):
            rows.append(row)
            cols.append(n)
            data.append(-value if sign == 1 else value)
        i_lambda = i_lp if sign == 1 else i_lm
        if atom:
            i_slack = i_ap if sign == 1 else i_am
            target_value = 1.0
        else:
            i_slack = i_rp + j if sign == 1 else i_rm + j
            target_value = target_avg[j]
        rows.extend([row, row])
        cols.extend([i_lambda, i_slack])
        data.extend([-1.0, -1.0])
        rhs.append(-target_value if sign == 1 else target_value)
        row += 1

    add_support_constraint(1, atom=True)
    add_support_constraint(-1, atom=True)
    for j in range(bins):
        add_support_constraint(1, atom=False, j=j)
    for j in range(bins):
        add_support_constraint(-1, atom=False, j=j)

    a_ub = sparse.csr_matrix((data, (rows, cols)), shape=(row, n_var))
    objective = np.zeros(n_var)
    objective[[i_lp, i_lm, i_ap, i_am]] = 1.0
    objective[i_rp : i_rp + bins] = lipschitz * h
    objective[i_rm : i_rm + bins] = lipschitz * h
    bounds = [(None, None)] * m + [(0.0, None)] * (n_var - m)
    dual = linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.asarray(rhs),
        bounds=bounds,
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
        },
    )
    if not dual.success:
        raise RuntimeError(dual.message)

    # Primal variables c_f,c_g,q_f[0:K],q_g[0:K].  Maximization is
    # represented by minimizing the negative deployment separation.
    i_qf = 2
    i_qg = 2 + bins
    n_primal = 2 + 2 * bins
    primal_objective = np.zeros(n_primal)
    primal_objective[0] = -1.0
    primal_objective[1] = 1.0
    primal_objective[i_qf : i_qf + bins] = -target_int
    primal_objective[i_qg : i_qg + bins] = target_int

    a_eq = np.zeros((m, n_primal))
    a_eq[:, 0] = 1.0
    a_eq[:, 1] = -1.0
    a_eq[:, i_qf : i_qf + bins] = audit_int
    a_eq[:, i_qg : i_qg + bins] = -audit_int

    mass = np.zeros((2, n_primal))
    mass[0, 0] = 1.0
    mass[0, i_qf : i_qf + bins] = h
    mass[1, 1] = 1.0
    mass[1, i_qg : i_qg + bins] = h
    primal = linprog(
        primal_objective,
        A_ub=mass,
        b_ub=np.ones(2),
        A_eq=a_eq,
        b_eq=np.zeros(m),
        bounds=[(0.0, 1.0), (0.0, 1.0)]
        + [(0.0, lipschitz)] * (2 * bins),
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
        },
    )
    if not primal.success:
        raise RuntimeError(primal.message)

    return {
        "primal": -float(primal.fun),
        "discrete_dual": float(dual.fun),
        "a": np.asarray(dual.x[:m]),
        "lambda_plus_discrete": float(dual.x[i_lp]),
        "lambda_minus_discrete": float(dual.x[i_lm]),
    }


def residual_value(t, a: np.ndarray, n_target: int, sign: int = 1):
    t = np.asarray(t)
    value = np.full_like(t, 1.0 - float(np.sum(a)), dtype=float)
    for n, coefficient in enumerate(a, start=1):
        value += coefficient * t**n
    value -= t**n_target
    return sign * value


def residual_antiderivative(
    t, a: np.ndarray, n_target: int, threshold: float, sign: int
):
    t = np.asarray(t)
    value = (sign * (1.0 - float(np.sum(a))) - threshold) * t
    for n, coefficient in enumerate(a, start=1):
        value += sign * coefficient * t ** (n + 1) / (n + 1)
    value -= sign * t ** (n_target + 1) / (n_target + 1)
    return value


def positive_intervals(
    a: np.ndarray,
    n_target: int,
    threshold: float,
    sign: int,
    scan_points: int = 50_000,
):
    grid = np.linspace(0.0, 1.0, scan_points + 1)
    values = residual_value(grid, a, n_target, sign) - threshold
    crossing = np.flatnonzero(values[:-1] * values[1:] < 0.0)
    roots = [
        brentq(
            lambda x: float(residual_value(x, a, n_target, sign) - threshold),
            grid[j],
            grid[j + 1],
            xtol=2e-15,
        )
        for j in crossing
    ]
    points = np.asarray([0.0, *roots, 1.0])
    intervals = []
    for left, right in zip(points[:-1], points[1:]):
        midpoint = (left + right) / 2.0
        if residual_value(midpoint, a, n_target, sign) > threshold:
            intervals.append((left, right))
    return intervals


def continuum_support(
    a: np.ndarray, n_target: int, lipschitz: float, sign: int
):
    """Evaluate H_L(sign*z_a) through the scalar threshold formula."""
    coarse = np.linspace(0.0, 1.0, 100_001)
    values = residual_value(coarse, a, n_target, sign)
    z0 = float(residual_value(0.0, a, n_target, sign))
    # One is a safe extra margin for these bounded-target residual problems;
    # values above the true maximum cannot minimize the convex objective.
    lambda_max = max(1.0, float(np.max(values)) + 1.0, z0 + 1.0)

    def objective(threshold: float):
        intervals = positive_intervals(a, n_target, threshold, sign)
        positive_integral = sum(
            float(
                residual_antiderivative(
                    right, a, n_target, threshold, sign
                )
                - residual_antiderivative(
                    left, a, n_target, threshold, sign
                )
            )
            for left, right in intervals
        )
        return (
            threshold
            + max(z0 - threshold, 0.0)
            + lipschitz * positive_integral
        )

    cuts = [0.0]
    if 0.0 < z0 < lambda_max:
        cuts.append(z0)
    cuts.append(lambda_max)
    candidates = [(objective(0.0), 0.0)]
    for left, right in zip(cuts[:-1], cuts[1:]):
        if right - left > 1e-14:
            optimum = minimize_scalar(
                objective,
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-13, "maxiter": 250},
            )
            candidates.append((float(optimum.fun), float(optimum.x)))
        candidates.append((objective(right), right))
    return min(candidates)


def exact_frontier(m: int, n_target: int):
    angles = np.arange(1, m + 1) * np.pi / (2.0 * (m + 1))
    return 1.0 + 2.0 * np.sum(
        (-1.0) ** np.arange(1, m + 1) * np.cos(angles) ** (2 * n_target)
    )


def constructive_bounds(m: int, n_target: int, lipschitz: float):
    amplitude = min(1.0 / (m + 1), lipschitz / (m * (m + 1)))
    legendre_gap = np.prod(
        [(n_target - j) / (n_target + j) for j in range(1, m + 1)]
    )
    degree = (m - 1) // 2
    roots, _ = roots_legendre(degree + 1)
    largest_shifted = (roots[-1] + 1.0) / 2.0
    upper = min(
        exact_frontier(m, n_target),
        lipschitz * (1.0 - largest_shifted + 1.0 / (n_target + 1)),
        1.0,
    )
    return amplitude * legendre_gap, upper


def monotone_uniform_grid(m: int, n_target: int, points: int = 40_001):
    t = np.linspace(0.0, 1.0, points)
    basis = chebvander(2.0 * t - 1.0, m)
    target = t**n_target
    a_ub = np.vstack(
        [
            np.column_stack([basis, -np.ones(points)]),
            np.column_stack([-basis, -np.ones(points)]),
        ]
    )
    b_ub = np.concatenate([target, -target])
    result = linprog(
        np.concatenate([np.zeros(m + 1), [1.0]]),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * (m + 1) + [(0.0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return 2.0 * float(result.fun)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 512/1024 bins and a 1024-bin sensitivity run.",
    )
    args = parser.parse_args()
    convergence_bins = [512, 1024] if args.quick else [2048, 4096, 8192]
    sensitivity_bins = 1024 if args.quick else 4096
    cases = [(2, 8, 1.0), (4, 16, 1.0), (8, 64, 1.0), (8, 64, 4.0)]
    sensitivity_l = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    RESULTS.mkdir(exist_ok=True)
    check_rows = []
    coefficient_rows = []
    cache = {}
    for m, n_target, lipschitz in cases:
        previous = None
        for bins in convergence_bins:
            solved = solve_discrete(m, n_target, lipschitz, bins)
            cache[(m, n_target, lipschitz, bins)] = solved
            h_plus, lambda_plus = continuum_support(
                solved["a"], n_target, lipschitz, 1
            )
            h_minus, lambda_minus = continuum_support(
                solved["a"], n_target, lipschitz, -1
            )
            continuum_dual = h_plus + h_minus
            lower, upper = constructive_bounds(m, n_target, lipschitz)
            check_rows.append(
                {
                    "m": m,
                    "N": n_target,
                    "L": lipschitz,
                    "bins": bins,
                    "primal_lower": solved["primal"],
                    "discrete_dual": solved["discrete_dual"],
                    "discrete_gap": solved["discrete_dual"] - solved["primal"],
                    "continuum_dual_at_a": continuum_dual,
                    "continuum_minus_primal": continuum_dual - solved["primal"],
                    "change_from_previous_bins": ""
                    if previous is None
                    else solved["primal"] - previous,
                    "constructive_lower": lower,
                    "constructive_upper": upper,
                    "lambda_plus": lambda_plus,
                    "lambda_minus": lambda_minus,
                    "a_l1": float(np.sum(np.abs(solved["a"]))),
                }
            )
            previous = solved["primal"]
            if bins == convergence_bins[-1]:
                for n, coefficient in enumerate(solved["a"], start=1):
                    coefficient_rows.append(
                        {
                            "m": m,
                            "N": n_target,
                            "L": lipschitz,
                            "bins": bins,
                            "n": n,
                            "a_n": coefficient,
                        }
                    )

    sensitivity_rows = []
    for lipschitz in sensitivity_l:
        key = (8, 64, lipschitz, sensitivity_bins)
        solved = cache.get(key)
        if solved is None:
            solved = solve_discrete(8, 64, lipschitz, sensitivity_bins)
        h_plus, lambda_plus = continuum_support(solved["a"], 64, lipschitz, 1)
        h_minus, lambda_minus = continuum_support(solved["a"], 64, lipschitz, -1)
        lower, upper = constructive_bounds(8, 64, lipschitz)
        sensitivity_rows.append(
            {
                "m": 8,
                "N": 64,
                "L": lipschitz,
                "bins": sensitivity_bins,
                "primal_lower": solved["primal"],
                "continuum_dual_at_a": h_plus + h_minus,
                "constructive_lower": lower,
                "constructive_upper": upper,
                "lambda_plus": lambda_plus,
                "lambda_minus": lambda_minus,
                "a_l1": float(np.sum(np.abs(solved["a"]))),
            }
        )

    def write_csv(name: str, rows):
        with (RESULTS / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("capped_tail_dual_checks.csv", check_rows)
    write_csv("capped_tail_dual_coefficients.csv", coefficient_rows)
    write_csv("capped_tail_l_sensitivity.csv", sensitivity_rows)
    monotone_limit = monotone_uniform_grid(8, 64)
    summary = {
        "description": "Numerical evaluation of the exact capped-tail dual",
        "convergence_bins": convergence_bins,
        "sensitivity_bins": sensitivity_bins,
        "monotone_frontier_m8_N64_grid": monotone_limit,
        "max_final_discrete_primal_dual_gap": max(
            abs(row["discrete_gap"])
            for row in check_rows
            if row["bins"] == convergence_bins[-1]
        ),
        "max_final_continuum_primal_gap": max(
            row["continuum_minus_primal"]
            for row in check_rows
            if row["bins"] == convergence_bins[-1]
        ),
    }
    (RESULTS / "capped_tail_dual_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
