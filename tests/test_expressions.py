import numpy as np
import pytest

from plotxy_app.expressions import ExpressionError, collect_series_names, evaluate

A = np.array([1.0, 2.0, 3.0, 4.0])
B = np.array([10.0, 20.0, 30.0, 40.0])
SHORT = np.array([100.0, 200.0])
WEIRD = np.array([5.0, 6.0, 7.0, 8.0])  # referenced as "der(v)"

SERIES = {"A": A, "B": B, "short": SHORT, "der(v)": WEIRD}


def resolver(name):
    return SERIES[name]


def ev(expr):
    return evaluate(expr, resolver)


def test_operators():
    assert np.allclose(ev("A + B")[0], A + B)
    assert np.allclose(ev("A - B")[0], A - B)
    assert np.allclose(ev("A * B")[0], A * B)
    assert np.allclose(ev("B / A")[0], B / A)
    assert np.allclose(ev("A ** 2")[0], A ** 2)


def test_unary_and_parens():
    assert np.allclose(ev("-A")[0], -A)
    assert np.allclose(ev("+A")[0], A)
    assert np.allclose(ev("(A + B) * 2")[0], (A + B) * 2)
    assert np.allclose(ev("A + B * 2")[0], A + B * 2)  # precedence


def test_abs_and_constants():
    assert np.allclose(ev("abs(-A)")[0], A)
    assert np.allclose(ev("2 * A + 0.5")[0], 2 * A + 0.5)
    assert np.allclose(ev("2e-3 * A")[0], 2e-3 * A)


def test_sqrt():
    assert np.allclose(ev("sqrt(A)")[0], np.sqrt(A))
    assert np.allclose(ev("sqrt(A * B)")[0], np.sqrt(A * B))
    # sqrt of a negative yields nan (errstate ignores the warning)
    neg = {"n": np.array([-1.0, 4.0])}
    vals, _, _ = evaluate("sqrt(n)", lambda k: neg[k])
    assert np.isnan(vals[0]) and vals[1] == 2.0
    # sqrt is now a known function (no longer "desconhecida")
    assert collect_series_names("sqrt(A) + B") == {"A", "B"}


def test_string_reference():
    vals, names, _ = ev('"der(v)" + 1')
    assert np.allclose(vals, WEIRD + 1)
    assert names == {"der(v)"}


def test_collect_names():
    assert collect_series_names('A * "der(v)" - B') == {"A", "der(v)", "B"}


def test_truncation():
    vals, _, truncated = ev("A + short")
    assert truncated
    assert np.allclose(vals, A[:2] + SHORT)


def test_div_zero_and_nan():
    zero = {"z": np.array([0.0, 1.0]), "one": np.array([1.0, 1.0])}
    vals, _, _ = evaluate("one / z", lambda n: zero[n])
    assert np.isinf(vals[0]) and vals[1] == 1.0
    vals2, _, _ = evaluate("z / z", lambda n: zero[n])
    assert np.isnan(vals2[0])


def test_scalar_only_rejected():
    with pytest.raises(ExpressionError, match="pelo menos uma série"):
        ev("2 + 2")


def test_unknown_series():
    with pytest.raises(ExpressionError, match="Série não encontrada"):
        ev("nope + 1")


def test_security_rejections():
    for bad in [
        "__import__('os').system('dir')",
        "A.__class__",
        "A[0]",
        "A > B",
        "A if B else A",
        "lambda: 1",
        "[a for a in A]",
        "abs(A, x=1)",
        "unknown_func(A)",
        "A; B",
        "True",
    ]:
        with pytest.raises(ExpressionError):
            ev(bad)


def test_empty_and_invalid():
    with pytest.raises(ExpressionError, match="Informe"):
        ev("   ")
    with pytest.raises(ExpressionError, match="inválida"):
        ev("A + ")
