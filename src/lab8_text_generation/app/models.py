from collections import deque, defaultdict
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Set, Tuple


Production = Tuple[str, ...]


@dataclass
class Grammar:
    """Right-linear grammar with readable pretty-printing and bounded generation."""

    start: str = "S"
    rules: Dict[str, List[Production]] = field(default_factory=lambda: defaultdict(list))

    def add_rule(self, lhs: str, rhs: Production) -> None:
        """Add a production if it is not already present."""
        if rhs not in self.rules[lhs]:
            self.rules[lhs].append(rhs)

    def nonterminals(self) -> Set[str]:
        return set(self.rules.keys())

    @staticmethod
    def _fmt_rhs(rhs: Production) -> str:
        if not rhs:
            return "ε"
        return " ".join(rhs)

    def pretty(self) -> str:
        """Render grammar as a compact multiline string."""
        lines: List[str] = []
        ordered_lhs = [self.start] + [lhs for lhs in self.rules if lhs != self.start]
        seen: Set[str] = set()
        for lhs in ordered_lhs:
            if lhs in seen or lhs not in self.rules:
                continue
            seen.add(lhs)
            rhss = " | ".join(self._fmt_rhs(rhs) for rhs in self.rules[lhs])
            lines.append(f"{lhs} -> {rhss}")
        return "\n".join(lines)

    def copy(self) -> "Grammar":
        clone = Grammar(start=self.start)
        for lhs, rhss in self.rules.items():
            for rhs in rhss:
                clone.add_rule(lhs, rhs)
        return clone

    def simplify(self) -> "Grammar":
        """Inline trivial symbols and remove unreachable rules."""
        rules: Dict[str, List[Production]] = {
            lhs: list(dict.fromkeys(rhss)) for lhs, rhss in self.rules.items()
        }

        def unique(seq: List[Production]) -> List[Production]:
            out: List[Production] = []
            for item in seq:
                if item not in out:
                    out.append(item)
            return out

        # Inline terminal-only symbols and one-step unit chains.
        changed = True
        while changed:
            changed = False

            terminal_only = {
                lhs
                for lhs, rhss in rules.items()
                if len(rhss) == 1 and all(tok not in rules for tok in rhss[0])
            }

            for lhs in list(rules.keys()):
                new_rhss: List[Production] = []
                for rhs in rules[lhs]:
                    expanded: List[str] = []
                    replaced = False
                    for tok in rhs:
                        if tok in terminal_only and tok in rules:
                            expanded.extend(rules[tok][0])
                            replaced = True
                        else:
                            expanded.append(tok)
                    new_rhss.append(tuple(expanded))
                    changed |= replaced
                rules[lhs] = unique(new_rhss)

            # Very small unit-production elimination.
            for lhs in list(rules.keys()):
                unit_targets = [
                    rhs[0]
                    for rhs in rules[lhs]
                    if len(rhs) == 1 and rhs[0] in rules and rhs[0] != lhs
                ]
                if not unit_targets:
                    continue

                new_rhss = [
                    rhs
                    for rhs in rules[lhs]
                    if not (len(rhs) == 1 and rhs[0] in rules and rhs[0] != lhs)
                ]
                for target in unit_targets:
                    for rhs in rules[target]:
                        if rhs not in new_rhss:
                            new_rhss.append(rhs)
                rules[lhs] = unique(new_rhss)
                changed = True

        # Remove unreachable symbols.
        reachable: Set[str] = {self.start}
        queue: deque[str] = deque([self.start])
        while queue:
            lhs = queue.popleft()
            for rhs in rules.get(lhs, []):
                for tok in rhs:
                    if tok in rules and tok not in reachable:
                        reachable.add(tok)
                        queue.append(tok)

        compact = Grammar(start=self.start)
        ordered = [self.start] + [lhs for lhs in rules if lhs != self.start]
        for lhs in ordered:
            if lhs not in reachable or lhs not in rules:
                continue
            for rhs in rules[lhs]:
                compact.add_rule(lhs, rhs)
        return compact

    def generate_strings(
        self,
        limit: int = 30,
        max_depth: int = 16,
        max_length: int = 60,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Generate terminal strings using bounded breadth-first expansion."""
        rng = random.Random(seed)
        queue: deque[Tuple[Tuple[str, ...], int]] = deque([((self.start,), 0)])
        seen: Set[str] = set()
        out: List[str] = []

        while queue and len(out) < limit:
            sentential, depth = queue.popleft()
            terminal_len = sum(len(tok) for tok in sentential if tok not in self.rules)
            if terminal_len > max_length:
                continue

            if all(tok not in self.rules for tok in sentential):
                candidate = "".join(sentential)
                if candidate not in seen:
                    seen.add(candidate)
                    out.append(candidate)
                continue

            if depth >= max_depth:
                continue

            # Expand the leftmost nonterminal.
            for idx, tok in enumerate(sentential):
                if tok in self.rules:
                    lhs = tok
                    break
            else:
                continue

            productions = list(self.rules[lhs])
            rng.shuffle(productions)
            for rhs in productions:
                next_sentential = sentential[:idx] + rhs + sentential[idx + 1 :]
                queue.append((next_sentential, depth + 1))

        return out


class SymbolFactory:
    """Generate compact nonterminal names: A, B, C ... Z, A2, B2 ..."""

    def __init__(self) -> None:
        self.index = 1

    def new(self) -> str:
        n = self.index
        self.index += 1
        letter = chr(ord("A") + (n - 1) % 26)
        suffix = "" if n <= 26 else str((n - 1) // 26 + 1)
        return f"{letter}{suffix}"


@dataclass(frozen=True)
class TerminalNode:
    text: str


@dataclass(frozen=True)
class RepeatNode:
    char: str
    tail: "PatternNode"


@dataclass(frozen=True)
class PrefixNode:
    prefix: str
    child: "PatternNode"


@dataclass(frozen=True)
class ChoiceNode:
    alts: Tuple[Tuple[str, "PatternNode"], ...]


PatternNode = TerminalNode | RepeatNode | PrefixNode | ChoiceNode


@dataclass
class PrefixTrieNode:
    children: Dict[str, "PrefixTrieNode"] = field(default_factory=dict)
    tails: Set[Production] = field(default_factory=set)
