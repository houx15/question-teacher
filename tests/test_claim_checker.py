import inspect

import pytest
from pydantic import ValidationError

from app.claim_checker import ClaimChecker, ClaimStatus
from app.schemas import GroundingCheckRequest


def request(payload):
    return GroundingCheckRequest.model_validate(payload)


def check_payload(
    *,
    kind="equivalence",
    expression="x",
    expected="x",
    substitutions=None,
    nonzero_symbols=None,
):
    return request(
        {
            "check_id": "check",
            "kind": kind,
            "expression": expression,
            "expected": expected,
            "substitutions": substitutions or {},
            "nonzero_symbols": nonzero_symbols or [],
            "conclusion_linked": True,
        }
    )


def test_substitution_with_parameters():
    check = check_payload(
        kind="substitution",
        expression="x^2-2*m*x+2*n",
        expected="4*n^2-4*m*n+2*n",
        substitutions={"x": "2*n"},
    )
    result = ClaimChecker().check(check)

    assert result.status == ClaimStatus.PASSED
    assert result.check_id == "check"
    assert result.conclusion_linked is True
    assert result.request_fingerprint == (
        ClaimChecker.request_fingerprint(check)
    )
    assert len(result.request_fingerprint) == 64


def test_request_fingerprint_binds_all_canonical_request_fields():
    checker = ClaimChecker()
    first = check_payload(expression="x", expected="x")
    second = check_payload(expression="m", expected="m")

    assert first.check_id == second.check_id
    assert checker.check(first).request_fingerprint != (
        checker.check(second).request_fingerprint
    )


def test_false_substitution_fails():
    result = ClaimChecker().check(
        check_payload(
            kind="substitution",
            expression="x^2-2*m*x+2*n",
            expected="4*n^2-2*m*n+2*n",
            substitutions={"x": "2*n"},
        )
    )

    assert result.status == ClaimStatus.FAILED


def test_equivalent_factorization():
    result = ClaimChecker().check(
        check_payload(
            expression="4*n^2-4*m*n+2*n",
            expected="2*n*(2*n-2*m+1)",
        )
    )

    assert result.status == ClaimStatus.PASSED


def test_nonzero_division_requires_declared_assumption():
    result = ClaimChecker().check(
        check_payload(
            kind="nonzero_division",
            expression="2*n*(2*n-2*m+1)",
            expected="2*n-2*m+1",
        )
    )

    assert result.status == ClaimStatus.UNSUPPORTED
    assert result.reason_code == "missing_nonzero_assumption"


def test_nonzero_division_passes_for_verified_declared_factor():
    result = ClaimChecker().check(
        check_payload(
            kind="nonzero_division",
            expression="2*n*(2*n-2*m+1)",
            expected="2*n-2*m+1",
            nonzero_symbols=["n"],
        )
    )

    assert result.status == ClaimStatus.PASSED


@pytest.mark.parametrize(
    ("expression", "expected", "nonzero_symbols"),
    [
        ("n+1", "1", ["n"]),
        ("(n+1)*(m-n)", "m-n", ["n"]),
        ("n*(m-n)", "m-n", ["m"]),
        ("0", "m-n", ["m", "n"]),
    ],
)
def test_nonzero_division_rejects_unverified_or_undeclared_factor(
    expression,
    expected,
    nonzero_symbols,
):
    result = ClaimChecker().check(
        check_payload(
            kind="nonzero_division",
            expression=expression,
            expected=expected,
            nonzero_symbols=nonzero_symbols,
        )
    )

    assert result.status == ClaimStatus.UNSUPPORTED


def test_nonzero_division_rejects_a_non_factor_transition():
    result = ClaimChecker().check(
        check_payload(
            kind="nonzero_division",
            expression="n+1",
            expected="n",
            nonzero_symbols=["n"],
        )
    )

    assert result.status == ClaimStatus.UNSUPPORTED


def test_back_substitution_can_verify_the_reported_conclusion():
    result = ClaimChecker().check(
        check_payload(
            kind="back_substitution",
            expression="2*n-2*m+1",
            expected="0",
            substitutions={"m": "n+1/2"},
            nonzero_symbols=["n"],
        )
    )

    assert result.status == ClaimStatus.PASSED


def test_false_back_substitution_is_a_reproducible_contradiction():
    result = ClaimChecker().check(
        check_payload(
            kind="back_substitution",
            expression="x^2-2*m*x+2*n",
            expected="0",
            substitutions={"x": "2*n", "m": "n+2"},
            nonzero_symbols=["n"],
        )
    )

    assert result.status == ClaimStatus.FAILED


def test_unicode_operators_are_normalized():
    result = ClaimChecker().check(
        check_payload(
            expression="2×n²−2÷2",
            expected="2*n^2-1",
        )
    )

    assert result.status == ClaimStatus.PASSED


def test_small_numeric_power_remains_supported():
    result = ClaimChecker().check(
        check_payload(expression="2^2", expected="4")
    )

    assert result.status == ClaimStatus.PASSED


@pytest.mark.parametrize(
    "nested_power",
    [
        "2^(2^2)",
        "(2^2)^2",
    ],
)
def test_nested_pure_numeric_power_is_rejected_before_evaluation(
    nested_power,
):
    result = ClaimChecker().check(
        check_payload(expression=nested_power, expected="16")
    )

    assert result.status == ClaimStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "hostile",
    [
        "__import__('os')",
        "x.__class__",
        "x[0]",
        "x;1",
        "sin(x)",
        "a+b+c+d+e",
        "x^999",
        "(" * 13 + "x" + ")" * 13,
    ],
)
def test_checker_rejects_unsafe_or_unbounded_expressions(hostile):
    if hostile in {"__import__('os')", "x.__class__"}:
        with pytest.raises(ValidationError):
            check_payload(expression=hostile, expected="0")
        return
    result = ClaimChecker().check(
        check_payload(
            expression=hostile,
            expected="0",
        )
    )

    assert result.status == ClaimStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "unbounded",
    [
        "1234567890123+x",
        "+".join(["x"] * 66),
        "x^-1",
        "x^(1/2)",
        "1e999999",
    ],
)
def test_checker_rejects_each_resource_limit(unbounded):
    if unbounded == "1e999999":
        with pytest.raises(ValidationError):
            check_payload(expression=unbounded, expected="0")
        return
    result = ClaimChecker().check(
        check_payload(expression=unbounded, expected="0")
    )

    assert result.status == ClaimStatus.UNSUPPORTED


def test_checker_enforces_its_expression_length_boundary_defensively():
    unchecked_request = GroundingCheckRequest.model_construct(
        check_id="check",
        kind="equivalence",
        expression="x" * 257,
        expected="0",
        substitutions={},
        nonzero_symbols=[],
        conclusion_linked=True,
    )

    result = ClaimChecker().check(unchecked_request)

    assert result.status == ClaimStatus.UNSUPPORTED


@pytest.mark.parametrize(
    "checker_error",
    [
        MemoryError("out of memory"),
        PermissionError("permission denied"),
    ],
)
def test_checker_propagates_system_parser_errors(
    monkeypatch,
    checker_error,
):
    def broken_parse_expr(*args, **kwargs):
        raise checker_error

    monkeypatch.setattr(
        "app.claim_checker.parse_expr",
        broken_parse_expr,
    )

    with pytest.raises(type(checker_error), match=str(checker_error)):
        ClaimChecker().check(check_payload())


def test_checker_converts_malformed_expression_to_unsupported():
    result = ClaimChecker().check(
        check_payload(expression="x+", expected="0")
    )

    assert result.status == ClaimStatus.UNSUPPORTED
    assert result.reason_code == "unsupported_expression"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected", "sin(x)"),
        ("substitution", "__import__('os')"),
        ("substitution", "a+b+c+d+e"),
    ],
)
def test_checker_applies_parser_bounds_to_every_expression(field, value):
    payload = {
        "kind": "substitution",
        "expression": "x",
        "expected": "0",
        "substitutions": {"x": "1"},
    }
    if field == "substitution":
        payload["substitutions"]["x"] = value
    else:
        payload[field] = value

    if value == "__import__('os')":
        with pytest.raises(ValidationError):
            check_payload(**payload)
        return

    result = ClaimChecker().check(check_payload(**payload))

    assert result.status == ClaimStatus.UNSUPPORTED


def test_checker_builds_real_symbols_internally():
    checker = ClaimChecker()
    expression = checker._parse_expression("m+n")

    assert {symbol.name for symbol in expression.free_symbols} == {"m", "n"}
    assert all(symbol.is_real is True for symbol in expression.free_symbols)


def test_checker_never_uses_python_eval_or_exec():
    source = inspect.getsource(ClaimChecker)
    assert "eval(" not in source
    assert "exec(" not in source


def test_checker_does_not_use_a_broad_exception_handler():
    source = inspect.getsource(ClaimChecker)

    assert "except Exception" not in source
