# The geometry of AI validation

Source, code, processed data, and deterministic outputs for:

> **The geometry of AI validation: Exact certification limits for iid best-of-N search**

The repository is the complete arXiv/Overleaf project. `main.tex` is the only
root document. It produces one PDF containing the two-column article followed
by the complete one-column Supporting Information from
`supplementary_material.tex`.

## Code availability

The maintained project is available at:

<https://github.com/rfitas-lab/geometry-of-ai-validation>

The repository contains the manuscript source, processed numeric inputs,
theorem checks, optimization outputs, empirical reconstruction scripts, and all
deterministic figure code. Large raw generations, programs, and execution logs
are not redistributed; the reconstruction scripts identify and verify the
pinned public sources.

## Compile the paper

The project uses standard packages included in a current TeX Live installation.
No journal-specific class or style file is required.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The supplied `main.bbl` allows arXiv to compile the project without running
BibTeX. `references.bib` is included for editing and independent rebuilding.

## Lightweight reproduction

The lightweight workflow uses only the included processed data, downloads
nothing, and makes no model API calls.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/reproduce_geometry_validation.py
python code/solve_capped_tail_dual.py
```

The reference environment used Python 3.12.13, NumPy 2.3.5, SciPy 1.17.0,
Matplotlib 3.10.8, and scikit-learn 1.8.0.

## Full empirical reconstruction

The GSM8K workflow downloads two raw files from the public Monkey Business
dataset pinned to revision
`a9f8f73bcd6948a57ed922cba4e48062ef95f553`. Expected SHA-256 digests are
embedded in the scripts.

```bash
python code/analyze_search_spectrum.py
python code/analyze_split_robustness.py
python code/reproduce_geometry_validation.py
```

The CodeRM/HumanEval+ workflow uses the CodeRM repository pinned to commit
`aa4946e9245ed41e24d60ad29e965132b5b84fe6` and its public execution archive:

```bash
python code/analyze_coderm_search.py \
  --annotations-dir /path/to/CodeRM/data/result/humaneval+ \
  --results-root /path/to/extracted/output/humaneval+
python code/reproduce_geometry_validation.py
```

## Project map

- `main.tex` - arXiv-style article and combined-document driver.
- `supplementary_material.tex` - full proofs, extensions, empirical methods,
  robustness checks, and contribution boundary.
- `references.bib` and `main.bbl` - editable and compiled bibliography.
- `code/` - reproduction and raw-data reconstruction scripts.
- `data/processed/` - processed inputs used by the lightweight workflow.
- `results/` - machine-readable theorem, optimization, planning, and empirical
  outputs.
- `figures/` - final vector and raster figures.
- `Geometry_of_AI_Validation_arXiv.pdf` - compiled article plus Supporting
  Information.

## Reproducibility anchors

- Analysis seed: `20260818`.
- Theorem verification: 32 `(m, N)` cases.
- Shape-restricted verification: 30 `(m, N)` cases at `L=1`.
- Maximum closed-form versus quadrature discrepancy: `1.887e-15`.
- Maximum audited-moment residual: `3.331e-16`.
- Maximum shape-restricted moment residual: `6.356e-17`.
- CodeRM replication: 164 tasks, 32,800 programs, and 3.28 million
  candidate-test executions.
- Frozen-rule held-out audit: 82 discovery and 82 held-out tasks, 5,000
  replays, and a 500-label evaluation budget.

## Scope

The exact closed form applies to iid candidates, scalar ranking, randomized
ties, maximum selection, bounded truth, and a stable rank-truth relation. The
general span theorem applies to other mechanisms only when their audit and
deployment kernels are known or independently estimated on a common stable
state space. The empirical analyses are retrospective mechanism studies, not
prospective interventions.
