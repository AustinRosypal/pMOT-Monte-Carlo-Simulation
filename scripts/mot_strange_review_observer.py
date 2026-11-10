"""Give independent tighter adaptive reviews a larger iteration budget.

The observation duration, force calls, integrator, event rules and convergence
tolerances come from the frozen observer. Only its adaptive loop bound changes.
Production continues to use the original observer and its original loop bound.
"""

import ast
from hashlib import sha256
import inspect

from numba import njit

from pmot.mot_strange import loading_scan_retention as retention


ADAPTIVE_REVIEW_ITERATIONS = 16_000_000


def observer_tree():
    original = inspect.getsource(retention.observe.py_func)
    tree = ast.parse(original)
    function = tree.body[0]
    assignments = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "max_steps"
                for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Frozen observer iteration assignment changed")
    bound = assignments[0].value
    if not isinstance(bound, ast.IfExp) or not isinstance(bound.orelse, ast.Constant):
        raise ValueError("Frozen observer iteration expression changed")
    if bound.orelse.value != 2_000_000:
        raise ValueError("Frozen observer adaptive loop bound changed")
    bound.orelse.value = ADAPTIVE_REVIEW_ITERATIONS
    function.decorator_list = []
    ast.fix_missing_locations(tree)
    proof = {
        "original_observer_source_sha256": sha256(original.encode()).hexdigest(),
        "review_observer_ast_sha256": sha256(ast.dump(tree).encode()).hexdigest(),
        "adaptive_iteration_budget": ADAPTIVE_REVIEW_ITERATIONS,
        "sole_algorithm_change": "adaptive iteration bound 2000000 to 16000000",
    }
    return tree, proof


def control_proof():
    return observer_tree()[1]


def make_observer():
    tree, _ = observer_tree()
    namespace = dict(retention.observe.py_func.__globals__)
    exec(compile(tree, __file__, "exec"), namespace)
    return njit(cache=False)(namespace["observe"])
