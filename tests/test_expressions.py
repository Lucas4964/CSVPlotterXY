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


def test_derivative_operator():
    X = np.array([0.0, 1.0, 2.0, 3.0])  # uniform spacing
    vals, names, _ = evaluate("D(A)", resolver, x=X)
    assert np.allclose(vals, np.gradient(A, X))
    assert names == {"A"}
    # non-uniform X
    Xnu = np.array([0.0, 0.5, 2.0, 5.0])
    vals, _, _ = evaluate("D(A)", resolver, x=Xnu)
    assert np.allclose(vals, np.gradient(A, Xnu))
    # derivative of a sub-expression
    vals, _, _ = evaluate("D(2 * A)", resolver, x=X)
    assert np.allclose(vals, np.gradient(2 * A, X))
    # D is a known function now
    assert collect_series_names("D(A) + B") == {"A", "B"}
    assert "D" in _known_from_error()


def _known_from_error():
    try:
        collect_series_names("nope_func(A)")
    except ExpressionError as e:
        return str(e)
    return ""


def test_derivative_errors():
    X = np.array([0.0, 1.0, 2.0, 3.0])
    # missing X axis
    with pytest.raises(ExpressionError, match="eixo X"):
        evaluate("D(A)", resolver)
    # wrong arity
    with pytest.raises(ExpressionError, match="1 argumento"):
        evaluate("D(A, B)", resolver, x=X)
    # X and series of different lengths -> truncated together
    Xlong = np.array([0.0, 1.0])
    vals, _, truncated = evaluate("D(A)", resolver, x=Xlong)
    assert truncated
    assert np.allclose(vals, np.gradient(A[:2], Xlong))


def test_trig_and_exp_functions():
    assert np.allclose(ev("sin(A)")[0], np.sin(A))
    assert np.allclose(ev("cos(A)")[0], np.cos(A))
    assert np.allclose(ev("tan(A)")[0], np.tan(A))
    assert np.allclose(ev("exp(A)")[0], np.exp(A))
    assert np.allclose(ev("log(A)")[0], np.log(A))
    assert np.allclose(ev("log10(A)")[0], np.log10(A))
    # composition with operators
    assert np.allclose(ev("2 * sin(A) + cos(B)")[0], 2 * np.sin(A) + np.cos(B))
    # log of a non-positive value yields nan/-inf under errstate
    neg = {"n": np.array([-1.0, 0.0, np.e])}
    vals, _, _ = evaluate("log(n)", lambda k: neg[k])
    assert np.isnan(vals[0]) and np.isinf(vals[1]) and abs(vals[2] - 1.0) < 1e-12
    # all are announced as known functions
    known = _known_from_error()
    for f in ("sin", "cos", "tan", "exp", "log", "log10"):
        assert f in known


def _cumtrapz(y, x):
    seg = np.diff(x) * (y[:-1] + y[1:]) / 2.0
    return np.concatenate(([0.0], np.cumsum(seg)))


def test_integral_operator():
    X = np.array([0.0, 1.0, 2.0, 3.0])
    vals, names, _ = evaluate("I(A)", resolver, x=X)
    assert np.allclose(vals, _cumtrapz(A, X))
    assert vals[0] == 0.0 and len(vals) == len(A)
    assert names == {"A"}
    # non-uniform X
    Xnu = np.array([0.0, 0.5, 2.0, 5.0])
    vals, _, _ = evaluate("I(A)", resolver, x=Xnu)
    assert np.allclose(vals, _cumtrapz(A, Xnu))
    # integral of a sub-expression, and D/I composition parses
    vals, _, _ = evaluate("I(2 * A)", resolver, x=X)
    assert np.allclose(vals, _cumtrapz(2 * A, X))
    # I(D(y)) recovers y up to a constant (fundamental theorem, trapezoid)
    vals, _, _ = evaluate("I(D(A))", resolver, x=X)
    assert np.allclose(vals + A[0], A)
    assert "I" in _known_from_error()


def test_integral_errors():
    X = np.array([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ExpressionError, match="eixo X"):
        evaluate("I(A)", resolver)
    with pytest.raises(ExpressionError, match="1 argumento"):
        evaluate("I(A, B)", resolver, x=X)


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
