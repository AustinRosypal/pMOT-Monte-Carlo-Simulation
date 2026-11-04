import ast
import importlib
from pathlib import Path
import numpy as np
import pytest
from numba import njit
from pmot.mot_multilevel import adaptive


@njit
def acceleration(r, v, axis, data):
    return np.zeros(3)


@njit
def field(r, data):
    return np.zeros(3)


@njit
def escaped(r, v, data):
    return False


@pytest.fixture
def budget(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    return importlib.import_module('mot_strange_retained_budget')


def test_equations_and_error_scales_identical_to_frozen_integrator(budget):
    root = Path(__file__).resolve().parents[2]
    def function(path, name):
        return next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
    old = function(root/'src/pmot/mot_multilevel/adaptive.py', 'integrate_local_environment')
    new = function(Path(budget.__file__), 'integrate_budgeted')
    new.name = old.name
    new.decorator_list = old.decorator_list = []
    new.args.args.pop()
    new.args.defaults.pop()
    for n in ast.walk(new):
        if isinstance(n, ast.For) and isinstance(n.iter, ast.Call):
            if isinstance(n.iter.func, ast.Name) and n.iter.func.id == 'range':
                if isinstance(n.iter.args[0], ast.Name) and n.iter.args[0].id == 'max_attempts':
                    n.iter.args[0] = ast.Constant(value=2000000)
    assert ast.dump(old) == ast.dump(new)


def arguments():
    return (np.array([.005, 0., 0.]), np.zeros(3), .002,
            tuple(np.zeros(7) for _ in range(11)), acceleration, field, escaped,
            2., True, adaptive.unlimited_step)


def test_larger_budget_completes_without_changing_solution(budget):
    with pytest.raises(ValueError, match='iteration limit'):
        budget.integrate_budgeted(*arguments(), 2)
    actual = budget.integrate_budgeted(*arguments(), 100)
    reference = adaptive.integrate_local_environment.py_func(*arguments())
    assert actual[0] == -1
    assert abs(actual[1]-.002) <= 1e-12
    for a, b in zip(actual, reference):
        np.testing.assert_array_equal(a, b)


def test_attempt_budget_remains_finite_and_explicit(budget):
    for limit in (0, 50000001, float('inf')):
        with pytest.raises(ValueError, match='finite attempt budget'):
            budget.retained_adaptive(None, None, 2., None, 512., limit)
