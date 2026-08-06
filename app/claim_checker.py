import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set, Tuple

from sympy import (
    Add,
    Expr,
    Float,
    Integer,
    Mul,
    Pow,
    Rational,
    Symbol,
    count_ops,
    fraction,
    preorder_traversal,
    simplify,
)
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    rationalize,
    standard_transformations,
)

from app.schemas import GroundingCheckRequest


class ClaimStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ClaimCheckResult:
    check_id: str
    status: ClaimStatus
    conclusion_linked: bool
    reason_code: str


class _UnsupportedClaim(ValueError):
    pass


class ClaimChecker:
    _MAX_EXPRESSION_LENGTH = 256
    _MAX_SYMBOLS = 4
    _MAX_NESTING = 12
    _MAX_DIGITS = 12
    _MAX_LITERAL_EXPONENT = 4
    _MAX_OPERATIONS = 64
    _CHARACTER_PATTERN = re.compile(r"[0-9A-Za-z+\-*/^().\s]+")
    _IDENTIFIER_PATTERN = re.compile(r"[A-Za-z]+")
    _SCIENTIFIC_NOTATION_PATTERN = re.compile(
        r"(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+"
    )
    _TRANSFORMATIONS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
        rationalize,
    )
    _NORMALIZATION_TABLE = str.maketrans(
        {
            "×": "*",
            "÷": "/",
            "−": "-",
            "－": "-",
            "–": "-",
            "—": "-",
            "＋": "+",
            "²": "^2",
        }
    )
    _GLOBAL_DICT = {
        "__builtins__": {},
        "Add": Add,
        "Integer": Integer,
        "Float": Float,
        "Mul": Mul,
        "Pow": Pow,
        "Rational": Rational,
    }

    def check(self, request: GroundingCheckRequest) -> ClaimCheckResult:
        symbols: Dict[str, Symbol] = {}
        try:
            expression = self._parse_expression(
                request.expression,
                symbols,
            )
            expected = self._parse_expression(
                request.expected,
                symbols,
            )
            substitutions = self._parse_substitutions(
                request.substitutions,
                symbols,
            )
            self._register_symbol_names(
                set(request.nonzero_symbols),
                symbols,
            )

            if request.kind == "substitution":
                actual = expression.subs(substitutions, simultaneous=True)
                passed = simplify(actual - expected) == 0
            elif request.kind == "equivalence":
                passed = simplify(expression - expected) == 0
            elif request.kind == "back_substitution":
                actual = expression.subs(substitutions, simultaneous=True)
                passed = simplify(actual - expected) == 0
            elif request.kind == "nonzero_division":
                return self._check_nonzero_division(
                    request,
                    expression,
                    expected,
                    symbols,
                )
            else:
                raise _UnsupportedClaim("unsupported_check_kind")
        except _UnsupportedClaim as exc:
            return self._result(
                request,
                ClaimStatus.UNSUPPORTED,
                str(exc),
            )
        except Exception:
            return self._result(
                request,
                ClaimStatus.UNSUPPORTED,
                "unsupported_expression",
            )

        return self._result(
            request,
            ClaimStatus.PASSED if passed else ClaimStatus.FAILED,
            "equivalent" if passed else "not_equivalent",
        )

    def _parse_expression(
        self,
        text: str,
        symbols: Optional[Dict[str, Symbol]] = None,
    ) -> Expr:
        symbol_table = symbols if symbols is not None else {}
        normalized = self._normalize_expression(text)
        identifiers = self._IDENTIFIER_PATTERN.findall(normalized)
        if any(len(identifier) != 1 for identifier in identifiers):
            raise _UnsupportedClaim("unsupported_identifier")
        self._register_symbol_names(set(identifiers), symbol_table)

        try:
            unevaluated = parse_expr(
                normalized,
                local_dict=symbol_table,
                global_dict=self._GLOBAL_DICT,
                transformations=self._TRANSFORMATIONS,
                evaluate=False,
            )
            self._validate_expression_shape(
                unevaluated,
                set(symbol_table.values()),
            )
            expression = parse_expr(
                normalized,
                local_dict=symbol_table,
                global_dict=self._GLOBAL_DICT,
                transformations=self._TRANSFORMATIONS,
                evaluate=True,
            )
            self._validate_expression_shape(
                expression,
                set(symbol_table.values()),
            )
        except _UnsupportedClaim:
            raise
        except Exception as exc:
            raise _UnsupportedClaim("unsupported_expression") from exc

        return expression

    def _parse_substitutions(
        self,
        substitutions: Dict[str, str],
        symbols: Dict[str, Symbol],
    ) -> Dict[Symbol, Expr]:
        self._register_symbol_names(set(substitutions), symbols)
        return {
            symbols[name]: self._parse_expression(value, symbols)
            for name, value in substitutions.items()
        }

    def _check_nonzero_division(
        self,
        request: GroundingCheckRequest,
        expression: Expr,
        expected: Expr,
        symbols: Dict[str, Symbol],
    ) -> ClaimCheckResult:
        if expression == 0 or expected == 0:
            return self._result(
                request,
                ClaimStatus.UNSUPPORTED,
                "unsupported_canceled_factor",
            )

        quotient = simplify(expression / expected)
        numerator, denominator = fraction(quotient)
        if denominator != 1 or simplify(expected * quotient - expression) != 0:
            return self._result(
                request,
                ClaimStatus.UNSUPPORTED,
                "unsupported_canceled_factor",
            )
        if quotient == 1 or not self._is_verifiable_nonzero_factor(quotient):
            return self._result(
                request,
                ClaimStatus.UNSUPPORTED,
                "unsupported_canceled_factor",
            )

        canceled_symbols = quotient.free_symbols
        declared = {
            symbols[name]
            for name in request.nonzero_symbols
        }
        if not canceled_symbols.issubset(declared):
            return self._result(
                request,
                ClaimStatus.UNSUPPORTED,
                "missing_nonzero_assumption",
            )
        return self._result(
            request,
            ClaimStatus.PASSED,
            "verified_nonzero_division",
        )

    def _is_verifiable_nonzero_factor(self, factor: Expr) -> bool:
        coefficient, dependent = factor.as_coeff_Mul()
        if coefficient == 0 or coefficient.is_real is not True:
            return False
        if not dependent.free_symbols:
            return dependent == 1

        powers = dependent.as_powers_dict()
        return all(
            isinstance(base, Symbol)
            and isinstance(exponent, Integer)
            and int(exponent) > 0
            for base, exponent in powers.items()
        )

    def _normalize_expression(self, text: str) -> str:
        if not isinstance(text, str):
            raise _UnsupportedClaim("unsupported_expression")
        normalized = text.translate(self._NORMALIZATION_TABLE).strip()
        if not normalized or len(normalized) > self._MAX_EXPRESSION_LENGTH:
            raise _UnsupportedClaim("unsupported_expression")
        if self._CHARACTER_PATTERN.fullmatch(normalized) is None:
            raise _UnsupportedClaim("unsupported_character")
        if re.search(r"\*\s*\*|/\s*/", normalized):
            raise _UnsupportedClaim("unsupported_operator")
        if self._SCIENTIFIC_NOTATION_PATTERN.search(normalized):
            raise _UnsupportedClaim("unsupported_number")

        without_numbers = re.sub(
            r"(?:\d+(?:\.\d*)?|\.\d+)",
            "",
            normalized,
        )
        if "." in without_numbers:
            raise _UnsupportedClaim("unsupported_character")
        if any(
            len(digits) > self._MAX_DIGITS
            for digits in re.findall(r"\d+", normalized)
        ):
            raise _UnsupportedClaim("unsupported_number")
        self._validate_parenthesis_nesting(normalized)
        self._validate_literal_exponents(normalized)
        return normalized

    def _validate_expression_shape(
        self,
        expression: object,
        allowed_symbols: Set[Symbol],
    ) -> None:
        if not isinstance(expression, Expr):
            raise _UnsupportedClaim("unsupported_expression")
        if expression.free_symbols - allowed_symbols:
            raise _UnsupportedClaim("unsupported_symbol")
        if count_ops(expression, visual=False) > self._MAX_OPERATIONS:
            raise _UnsupportedClaim("too_many_operations")

        allowed_types = (
            Add,
            Mul,
            Pow,
            Integer,
            Rational,
            Float,
            Symbol,
        )
        for node in preorder_traversal(expression):
            if not isinstance(node, allowed_types):
                raise _UnsupportedClaim("unsupported_expression_node")
            if not isinstance(node, Pow):
                continue
            if (
                not node.free_symbols
                and (node.base.has(Pow) or node.exp.has(Pow))
            ):
                raise _UnsupportedClaim("numeric_complexity_exceeded")
            exponent = node.exp
            if node.base.free_symbols and exponent.is_negative:
                raise _UnsupportedClaim("unsupported_exponent")
            if isinstance(exponent, Integer):
                if abs(int(exponent)) > self._MAX_LITERAL_EXPONENT:
                    raise _UnsupportedClaim("unsupported_exponent")
            elif isinstance(exponent, Rational):
                if node.base.free_symbols:
                    raise _UnsupportedClaim("unsupported_exponent")
            else:
                raise _UnsupportedClaim("unsupported_exponent")
        if expression.is_real is not True:
            raise _UnsupportedClaim("non_real_expression")

    def _register_symbol_names(
        self,
        names: Set[str],
        symbols: Dict[str, Symbol],
    ) -> None:
        if any(re.fullmatch(r"[A-Za-z]", name) is None for name in names):
            raise _UnsupportedClaim("unsupported_identifier")
        if len(set(symbols).union(names)) > self._MAX_SYMBOLS:
            raise _UnsupportedClaim("too_many_symbols")
        for name in names:
            symbols.setdefault(name, Symbol(name, real=True))

    def _validate_parenthesis_nesting(self, text: str) -> None:
        nesting = 0
        for character in text:
            if character == "(":
                nesting += 1
                if nesting > self._MAX_NESTING:
                    raise _UnsupportedClaim("too_much_nesting")
            elif character == ")":
                nesting -= 1
                if nesting < 0:
                    raise _UnsupportedClaim("unbalanced_parentheses")
        if nesting != 0:
            raise _UnsupportedClaim("unbalanced_parentheses")

    def _validate_literal_exponents(self, text: str) -> None:
        for match in re.finditer(r"\^\s*([+-]?)\s*(\d+)", text):
            sign, digits = match.groups()
            value = int(f"{sign}{digits}")
            if abs(value) > self._MAX_LITERAL_EXPONENT:
                raise _UnsupportedClaim("unsupported_exponent")

    @staticmethod
    def _result(
        request: GroundingCheckRequest,
        status: ClaimStatus,
        reason_code: str,
    ) -> ClaimCheckResult:
        return ClaimCheckResult(
            check_id=request.check_id,
            status=status,
            conclusion_linked=request.conclusion_linked,
            reason_code=reason_code,
        )
