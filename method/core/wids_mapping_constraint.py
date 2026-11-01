"""The apache_3j diagnosis -> bodysystem mapping constraint as a native
TabularBench relation AST (single source of truth for our attack, CAA,
MOEVA and the official checker).

Encoding (interval semantics, existing grammar nodes ONLY — no backend or
grammar changes needed by any consumer):

    Or_{k in 1..11} And( Equal(s, k, tol=1e-6),
                         Or_{(lo,hi) in runs_k} And(lo - W <= d, d <= hi + W) )

- runs_k: the contiguous code runs of body system k in the observed data
  (reference/wids_apache3j_intervals.json; cross-system interleaving at
  <=0.02 granularity is 0, verified during extraction).
- W = 1e-4 raw units: absorbs float32 round-trip noise on two-decimal codes
  (ulp <= 2.4e-6 at 2201) while being 200x smaller than the smallest gap
  between runs — no semantic loosening, no zero-tolerance trap on f32
  submissions (the lcld lesson).
- s (label-encoded cat) is exact in float64; tolerance 1e-6 absorbs ulp.

The executor semantics (numpy + pytorch): <= -> hinge, = -> |l-r|-tol,
And -> sum, Or -> min. Penalty is 0 iff (d, s) is consistent.
"""
from __future__ import annotations

import json
import os

from tabularbench.constraints.relation_constraint import (
    AndConstraint,
    Constant,
    EqualConstraint,
    Feature,
    LessEqualConstraint,
    OrConstraint,
)

REF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "reference", "wids_apache3j_intervals.json")
DIAG = "apache_3j_diagnosis"
SYSTEM = "apache_3j_bodysystem"
WIDEN = 1e-4          # f32-noise margin on interval endpoints
S_TOL = 1e-6          # Equal(s, k) tolerance


def load_intervals(path=REF_PATH):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): [tuple(v) for v in runs] for k, runs in raw.items()}


def apache_mapping_constraint(path=REF_PATH, widen=WIDEN, s_tol=S_TOL):
    """Build the AST. 11 branches x (1 equality + interval disjunction)."""
    intervals = load_intervals(path)
    branches = []
    for k in sorted(intervals):
        runs = []
        for lo, hi in intervals[k]:
            runs.append(AndConstraint([
                LessEqualConstraint(Constant(lo - widen), Feature(DIAG)),
                LessEqualConstraint(Feature(DIAG), Constant(hi + widen)),
            ]))
        d_in = runs[0] if len(runs) == 1 else OrConstraint(runs)
        branches.append(AndConstraint([
            EqualConstraint(Feature(SYSTEM), Constant(float(k)),
                            tolerance=Constant(s_tol)),
            d_in,
        ]))
    return branches[0] if len(branches) == 1 else OrConstraint(branches)


def extended_relation_constraints(cons, path=REF_PATH):
    """31 native order pairs + the mapping constraint (the injection used
    by every consumer: our BatchJudge, ConstraintChecker, CAA, MOEVA)."""
    return list(cons.relation_constraints) + [apache_mapping_constraint(path)]
