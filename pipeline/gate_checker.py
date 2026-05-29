"""Gate evaluation for pipeline phase transitions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GateCondition:
    """One precondition that must be satisfied to pass a gate."""

    name: str
    check: Callable[[], tuple[bool, str]]
    remediation: str = ""


@dataclass
class GateResult:
    """Outcome of evaluating a gate's conditions."""

    gate_name: str
    passed: bool
    conditions: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)


class GateChecker:
    """Evaluate named gates with precondition lists for phase transitions."""

    def __init__(self) -> None:
        self._gates: dict[str, list[GateCondition]] = {}
        self._results: list[GateResult] = []

    def register(self, gate_name: str, conditions: list[GateCondition]) -> None:
        """Register a named gate with its list of preconditions."""

        self._gates[gate_name] = conditions

    def evaluate(self, gate_name: str) -> GateResult:
        """Evaluate all conditions for a gate and return the result."""

        conditions = self._gates.get(gate_name, [])
        evaluated: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        for condition in conditions:
            try:
                passed, message = condition.check()
            except Exception as exc:  # noqa: BLE001
                passed = False
                message = f"Condition check raised: {exc}"
                logger.warning("Gate condition '%s' raised an exception: %s", condition.name, exc)

            entry = {
                "name": condition.name,
                "passed": passed,
                "message": message,
                "remediation": condition.remediation,
            }
            evaluated.append(entry)
            if not passed:
                failures.append(entry)

        result = GateResult(
            gate_name=gate_name,
            passed=len(failures) == 0,
            conditions=evaluated,
            failures=failures,
        )
        self._results.append(result)

        if result.passed:
            logger.info("Gate '%s' PASSED (%s conditions checked).", gate_name, len(evaluated))
        else:
            logger.error(
                "Gate '%s' FAILED — %s of %s conditions unchecked: %s",
                gate_name,
                len(failures),
                len(evaluated),
                ", ".join(f["name"] for f in failures),
            )

        return result

    @property
    def all_results(self) -> list[GateResult]:
        return list(self._results)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all evaluated gates."""

        return {
            "gates_evaluated": len(self._results),
            "gates_passed": sum(1 for r in self._results if r.passed),
            "gates_failed": sum(1 for r in self._results if not r.passed),
            "details": [
                {
                    "gate": r.gate_name,
                    "passed": r.passed,
                    "failures": [f["name"] for f in r.failures],
                }
                for r in self._results
            ],
        }
