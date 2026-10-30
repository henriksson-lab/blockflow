# SPDX-License-Identifier: MIT
"""Solve an existing Python-MIP tracking problem with SCIP, then audit it.

The MPS transfer preserves the constraint matrix. The original objective is
reinstated explicitly, avoiding MPS objective-sense conventions. Every returned
value is checked against the original bounds, integrality and constraints.
"""
import math
from pathlib import Path
import time


def solve_with_scip(original, path, timeout=120, relative_gap=.01, threads=1, warm_start=None, verbose=False):
    from pyscipopt import Model, quicksum
    path = Path(path)
    original.write(str(path))
    if not path.exists():
        alternatives = list(path.parent.glob(path.name + "*"))
        if len(alternatives) != 1:
            raise RuntimeError("could not identify exported MPS problem")
        path = alternatives[0]
    model = Model("blockflow-mip-bridge")
    model.hideOutput(not verbose)
    model.readProblem(str(path))
    variables = {v.name: v for v in model.getVars()}
    original_names = {v.name for v in original.vars}
    if set(variables) != original_names or len(original_names) != len(original.vars):
        raise ValueError("MPS transfer changed variable identities")
    model.setObjective(quicksum(float(v.obj) * variables[v.name] for v in original.vars) + original.objective.const,
                       sense="maximize" if str(original.sense).upper() == "MAX" else "minimize")
    model.setRealParam("limits/time", timeout)
    model.setRealParam("limits/gap", relative_gap)
    model.setIntParam("parallel/maxnthreads", threads)
    seed_accepted = None
    seed_objective = None
    if warm_start is not None:
        audit_original_values(original, warm_start)
        seed = model.createSol()
        for name, variable in variables.items():
            model.setSolVal(seed, variable, warm_start[name])
        seed_objective = original.objective.const + sum(float(v.obj) * warm_start[v.name] for v in original.vars)
        seed_accepted = bool(model.addSol(seed, free=True))
        if not seed_accepted:
            raise ValueError("SCIP rejected the audited feasible warm start")
    start = time.monotonic()
    model.optimize()
    elapsed = time.monotonic() - start
    solution = model.getBestSol()
    if solution is None:
        raise RuntimeError(f"SCIP returned no feasible incumbent: {model.getStatus()}")
    values = {name: float(model.getSolVal(solution, variable)) for name, variable in variables.items()}
    worst = audit_original_values(original, values)
    objective = original.objective.const + sum(float(v.obj) * values[v.name] for v in original.vars)
    if not math.isclose(objective, model.getSolObjVal(solution), rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError("objective changed during MPS transfer")
    report = {"backend": "SCIP/PySCIPOpt", "status": str(model.getStatus()), "objective": objective,
              "bound": model.getDualbound(), "gap": model.getGap(), "solutions": model.getNSols(),
              "timeout_seconds": timeout, "optimize_seconds": elapsed, "model_file": str(path),
              "variables": len(values), "constraints": len(original.constrs),
              "original_model_max_violation": worst, "original_objective_verified": True,
              "warm_start_accepted": seed_accepted, "warm_start_objective": seed_objective,
              "warm_start_changed_variables": None if warm_start is None else sum(abs(values[name]-warm_start[name]) > 1e-6 for name in values)}
    return values, report


def audit_original_values(original, values):
    if set(values) != {v.name for v in original.vars} or any(not math.isfinite(value) for value in values.values()):
        raise ValueError("decision vector must have one finite value per original variable")
    worst = 0.0
    for variable in original.vars:
        value = values[variable.name]
        worst = max(worst, variable.lb - value, value - variable.ub)
        if variable.var_type != "C":
            worst = max(worst, abs(value - round(value)))
    for constraint in original.constrs:
        expression = constraint.expr
        value = expression.const + sum(coefficient * values[variable.name] for variable, coefficient in expression.expr.items())
        if expression.sense == "=":
            violation = abs(value)
        elif expression.sense == "<":
            violation = max(0.0, value)
        elif expression.sense == ">":
            violation = max(0.0, -value)
        else:
            raise ValueError("unrecognized original constraint sense")
        worst = max(worst, violation)
    if worst > 1e-6:
        raise ValueError(f"SCIP incumbent violates original MIP constraints: {worst}")
    return worst


def ultrack_leaf_seed(native, database_file):
    """Nonoverlapping leaf hypotheses plus greedy, nondividing temporal edges.

    This is only a feasible primal seed. It does not add constraints or change
    objective coefficients; the optimizer can replace every seeded decision.
    """
    import sqlite3
    with sqlite3.connect("file:" + str(database_file) + "?mode=ro", uri=True) as database:
        ancestors = {int(row[0]) for row in database.execute("SELECT DISTINCT ancestor_id FROM overlaps")}
    leaves = {i for i, identity in enumerate(native._backward_map) if int(identity) not in ancestors}
    pairs = native._edges_df.to_numpy()
    sources, targets, chosen = set(), set(), []
    order = sorted(range(len(native._edges)), key=lambda i: (-native._edges[i].obj, int(pairs[i, 0]), int(pairs[i, 1])))
    for i in order:
        a, b = map(int, pairs[i])
        if native._edges[i].obj <= .01 or a not in leaves or b not in leaves or a in sources or b in targets:
            continue
        sources.add(a)
        targets.add(b)
        chosen.append(i)
    selected = sources | targets
    if not selected:
        raise ValueError("leaf graph did not supply a nonempty feasible seed")
    values = {variable.name: 0.0 for variable in native._model.vars}
    for i in selected:
        values[native._nodes[i].name] = 1.0
        if i not in targets:
            values[native._appearances[i].name] = 1.0
        if i not in sources:
            values[native._disappearances[i].name] = 1.0
    for i in chosen:
        values[native._edges[i].name] = 1.0
    audit_original_values(native._model, values)
    return values, {"method": "nonoverlapping leaves plus greedy one-to-one edges with objective weight > .01",
                    "selected_nodes": len(selected), "selected_edges": len(chosen), "all_original_constraints_verified": True}


def ultrack_selected_dataframe(native, values):
    """Version-pinned conversion of ULtrack 0.8 decision arrays to node parents."""
    import numpy as np
    import pandas as pd
    ids = [int(native._backward_map[i]) for i, variable in enumerate(native._nodes) if values[variable.name] > .5]
    result = pd.DataFrame({"parent_id": -1}, index=ids)
    edge_indices = [i for i, variable in enumerate(native._edges) if values[variable.name] > .5]
    chosen = native._backward_map[native._edges_df.loc[edge_indices].to_numpy()]
    seen = set()
    for source, target in chosen:
        source, target = int(source), int(target)
        if source not in result.index or target not in result.index or target in seen:
            raise ValueError("selected Ultrack links have invalid node selection/parents")
        seen.add(target)
        result.loc[target, "parent_id"] = source
    if not ids:
        raise ValueError("SCIP selected no Ultrack segments")
    result.index = np.asarray(result.index, dtype="i8")
    return result
