import random

from .grammar import DEFAULT_GRAMMAR, Grammar


class ExampleGenerator:
    def __init__(self, grammar: Grammar = DEFAULT_GRAMMAR) -> None:
        self.grammar = grammar
        self.by_lhs = grammar.productions_by_lhs

    def generate(self, start: str, target_length: int = 12, seed: int | None = None) -> str:
        rng = random.Random(seed)
        tokens = self._expand(start, target_length=target_length, depth=0, rng=rng)
        result = "".join(tokens)
        if not (10 <= len(result) <= 15):
            # fall back to a safe, short derivation if random choice overshoots
            tokens = self._expand(start, target_length=10, depth=0, rng=rng)
            result = "".join(tokens)
        return result

    def _expand(
        self,
        symbol: str,
        target_length: int,
        depth: int,
        rng: random.Random,
    ) -> list[str]:
        if symbol in self.grammar.terminals:
            return [symbol]
        options = self.by_lhs.get(symbol, [])
        if not options:
            return []
        # Bias towards terminal rules as depth grows.
        terminal_rules = [p for p in options if len(p.rhs) == 1 and p.rhs[0] in self.grammar.terminals]
        recursive_rules = [p for p in options if p not in terminal_rules]
        if depth > 4 and terminal_rules:
            chosen = rng.choice(terminal_rules)
        elif recursive_rules and rng.random() < 0.6:
            chosen = rng.choice(recursive_rules)
        else:
            chosen = rng.choice(options)

        result: list[str] = []
        for part in chosen.rhs:
            result.extend(self._expand(part, target_length, depth + 1, rng))
        return result
