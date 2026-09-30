"""Immutable scalar facts for a forward may-taint analysis."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaintState:
    """Local variable names that may currently hold tainted scalar values."""

    tainted_variables: frozenset[str] = frozenset()

    def is_tainted(self, variable: str) -> bool:
        return variable in self.tainted_variables

    def taint(self, variable: str) -> "TaintState":
        return TaintState(self.tainted_variables | {variable})

    def clean(self, variable: str) -> "TaintState":
        return TaintState(self.tainted_variables - {variable})

    def join(self, other: "TaintState") -> "TaintState":
        """Keep taint from either incoming path."""
        return TaintState(self.tainted_variables | other.tainted_variables)
