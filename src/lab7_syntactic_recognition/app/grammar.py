"""Grammar definition for laboratory work 7.

The grammar is based on the methodology examples:
- two start symbols: S and T
- terminal alphabet: a, b, c, d, e
- internal symbols: Arm, Side, PairArm, LeftPart, RightPart, Base

The parser works with a bottom-up chart, so the grammar is stored as
plain production rules with terminals and non-terminals mixed in RHS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Production:
    lhs: str
    rhs: tuple[str, ...]


@dataclass(frozen=True)
class Grammar:
    terminals: frozenset[str]
    nonterminals: frozenset[str]
    start_symbols: tuple[str, ...]
    productions: tuple[Production, ...]

    @property
    def productions_by_lhs(self) -> dict[str, list[Production]]:
        grouped: dict[str, list[Production]] = {}
        for production in self.productions:
            grouped.setdefault(production.lhs, []).append(production)
        return grouped


def build_default_grammar() -> Grammar:
    terminals = frozenset({"a", "b", "c", "d", "e"})
    nonterminals = frozenset(
        {
            "S",
            "T",
            "Arm",
            "Side",
            "PairArm",
            "LeftPart",
            "RightPart",
            "Base",
        }
    )
    productions = (
        Production("S", ("PairArm", "PairArm")),
        Production("T", ("Base", "PairArm")),

        Production("PairArm", ("Side", "PairArm")),
        Production("PairArm", ("PairArm", "Side")),
        Production("PairArm", ("Arm", "RightPart")),
        Production("PairArm", ("LeftPart", "Arm")),

        Production("LeftPart", ("Arm", "c")),
        Production("RightPart", ("c", "Arm")),

        Production("Base", ("b", "Base")),
        Production("Base", ("Base", "b")),
        Production("Base", ("e",)),

        Production("Side", ("b", "Side")),
        Production("Side", ("Side", "b")),
        Production("Side", ("b",)),
        Production("Side", ("d",)),

        Production("Arm", ("b", "Arm")),
        Production("Arm", ("Arm", "b")),
        Production("Arm", ("a",)),
    )
    return Grammar(terminals, nonterminals, ("S", "T"), productions)


DEFAULT_GRAMMAR = build_default_grammar()
