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


def test_derivative_with_duplicated_x():
    # Modelica/ATP event points: two samples at the same instant
    X = np.array([0.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    Y = np.array([0.0, 2.0, 4.0, 10.0, 12.0, 14.0])  # jump at x=2
    res = {"j": Y}
    vals, _, _ = evaluate("D(j)", lambda k: res[k], x=X)
    assert np.all(np.isfinite(vals))          # no inf/NaN poisoning
    # each side of the jump uses its own segment (slope 2 everywhere)
    assert np.allclose(vals, 2.0)

    # a continuous sine sampled with duplicated instants still gives
    # a clean derivative ~ w*cos
    w = 2 * np.pi * 5.0
    t = np.linspace(0.0, 1.0, 2001)
    t = np.insert(t, [500, 1200], [t[500], t[1200]])  # duplicate 2 points
    y = np.sin(w * t)
    vals, _, _ = evaluate("D(s)", lambda k: {"s": y}[k], x=t)
    assert np.all(np.isfinite(vals))
    assert np.max(np.abs(vals - w * np.cos(w * t))) < w * 0.01

    # x repeated 3+ times: the isolated middle sample has no derivative
    X3 = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    Y3 = np.array([0.0, 1.0, 5.0, 9.0, 10.0])
    vals, _, _ = evaluate("D(a3)", lambda k: {"a3": Y3}[k], x=X3)
    assert np.isnan(vals[2]) and np.isfinite(vals[[0, 1, 3, 4]]).all()


def test_integral_with_duplicated_x_and_nan():
    # a jump at duplicated x adds no area
    X = np.array([0.0, 1.0, 1.0, 2.0])
    Y = np.array([1.0, 1.0, 5.0, 5.0])
    vals, _, _ = evaluate("I(step)", lambda k: {"step": Y}[k], x=X)
    assert np.allclose(vals, [0.0, 1.0, 1.0, 6.0])
    # one NaN sample must not wipe the rest of the integral
    X2 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    Y2 = np.array([1.0, 1.0, np.nan, 1.0, 1.0])
    vals, _, _ = evaluate("I(g)", lambda k: {"g": Y2}[k], x=X2)
    assert np.isfinite(vals).all()
    assert vals[-1] == 2.0   # the two NaN-adjacent segments contribute 0


def test_derivative_integral_sampling_intervals():
    # checklist: consistent results across sampling rates (uniform and not)
    for n in (101, 10001):
        t = np.linspace(0.0, 1.0, n)
        y = t ** 2
        d, _, _ = evaluate("D(q)", lambda k: {"q": y}[k], x=t)
        assert np.max(np.abs(d - 2 * t)) < 2e-2
        i, _, _ = evaluate("I(q)", lambda k: {"q": y}[k], x=t)
        assert np.max(np.abs(i - t ** 3 / 3)) < 2e-2
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 1.0, 4001))
    y = 3.0 * t
    d, _, _ = evaluate("D(r)", lambda k: {"r": y}[k], x=t)
    assert np.allclose(d, 3.0)
    i, _, _ = evaluate("I(r)", lambda k: {"r": y}[k], x=t)
    assert np.max(np.abs(i - 1.5 * (t ** 2 - t[0] ** 2))) < 1e-6


def test_generators_linspace_arange():
    vals, names, truncated = ev("linspace(0, 10, 11)")
    assert np.allclose(vals, np.linspace(0, 10, 11))
    assert names == set() and not truncated
    assert np.allclose(ev("linspace(0, 1)")[0], np.linspace(0, 1, 50))
    assert np.allclose(ev("arange(5)")[0], np.arange(5.0))
    assert np.allclose(ev("arange(1, 4)")[0], np.arange(1.0, 4.0))
    assert np.allclose(ev("arange(0, 10, 0.5)")[0], np.arange(0, 10, 0.5))
    # composition with functions, operators and other generators
    assert np.allclose(ev("sin(linspace(0, 6.28, 100))")[0],
                       np.sin(np.linspace(0, 6.28, 100)))
    assert np.allclose(ev("linspace(0, 1, 10) + arange(10)")[0],
                       np.linspace(0, 1, 10) + np.arange(10.0))
    assert np.allclose(ev("2 * linspace(0, 1, 5) + 1")[0],
                       2 * np.linspace(0, 1, 5) + 1)
    # same-length series mix works; D() over a generator works
    assert np.allclose(ev("A + linspace(0, 1, 4)")[0],
                       A + np.linspace(0, 1, 4))
    X = np.array([0.0, 1.0, 2.0, 3.0])
    vals, _, _ = evaluate("D(linspace(0, 9, 4))", resolver, x=X)
    assert np.allclose(vals, 3.0)
    # announced as known functions
    known = _known_from_error()
    assert "linspace" in known and "arange" in known


def test_generator_errors():
    with pytest.raises(ExpressionError, match="apenas números"):
        ev("linspace(A, 1, 10)")
    with pytest.raises(ExpressionError, match="inteiro"):
        ev("linspace(0, 1, 2.5)")
    with pytest.raises(ExpressionError, match="inteiro"):
        ev("linspace(0, 1, 1)")
    with pytest.raises(ExpressionError, match="passo"):
        ev("arange(0, 1, 0)")
    with pytest.raises(ExpressionError, match="máximo"):
        ev("arange(0, 1e12)")
    with pytest.raises(ExpressionError, match="intervalo vazio"):
        ev("arange(10, 0, 1)")
    with pytest.raises(ExpressionError, match="2 ou 3 argumentos"):
        ev("linspace(1)")
    with pytest.raises(ExpressionError, match="Tamanhos incompatíveis"):
        ev("A + linspace(0, 1, 10)")   # A has 4 points
    # scalar-only expressions are still rejected
    with pytest.raises(ExpressionError, match="pelo menos uma série"):
        ev("2 + 2")


def test_scalar_reductions():
    assert np.allclose(ev("A / max(A)")[0], A / A.max())
    assert np.allclose(ev("A - mean(A)")[0], A - A.mean())
    assert np.allclose(ev("A - min(A)")[0], A - A.min())
    assert np.allclose(ev("A / rms(A)")[0], A / np.sqrt(np.mean(A ** 2)))
    # reductions ignore NaN samples
    nan = {"g": np.array([1.0, np.nan, 3.0])}
    vals, _, _ = evaluate("g / max(g)", lambda k: nan[k])
    assert np.isclose(vals[0], 1.0 / 3.0) and np.isclose(vals[2], 1.0)
    # all announced as known
    known = _known_from_error()
    for f in ("max", "min", "mean", "rms", "shift"):
        assert f in known


def test_scalar_only_result_rejected():
    with pytest.raises(ExpressionError, match="não resultou em uma série"):
        ev("max(A)")


def test_function_arity_enforced():
    with pytest.raises(ExpressionError, match="1 argumento"):
        ev("max(A, B)")
    # abs(A, B) would be np.abs(A, out=B) and silently overwrite B's data
    with pytest.raises(ExpressionError, match="1 argumento"):
        ev("abs(A, B)")
    with pytest.raises(ExpressionError, match="2 argumentos"):
        ev("shift(A)")


def test_shift():
    vals, _, _ = ev("shift(A, 1)")
    assert np.isnan(vals[0]) and np.allclose(vals[1:], A[:-1])
    vals, _, _ = ev("shift(A, -1)")
    assert np.isnan(vals[-1]) and np.allclose(vals[:-1], A[1:])
    assert np.allclose(ev("shift(A, 0)")[0], A)
    vals, _, _ = ev("shift(A, 10)")     # |n| >= len -> all NaN
    assert np.isnan(vals).all()
    # composition: first difference via shift
    vals, _, _ = ev("A - shift(A, 1)")
    assert np.isnan(vals[0]) and np.allclose(vals[1:], np.diff(A))
    # accepts 2.0, rejects 2.5 and series as n
    assert np.allclose(ev("shift(A, 2.0)")[0][2:], A[:-2])
    with pytest.raises(ExpressionError, match="inteiro"):
        ev("shift(A, 2.5)")
    with pytest.raises(ExpressionError, match="inteiro"):
        ev("shift(A, B)")


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
