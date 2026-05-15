from dataclasses import dataclass, field

from .grammar import Grammar, Production, DEFAULT_GRAMMAR


@dataclass
class ParseNode:
    symbol: str
    span: tuple[int, int]
    children: list["ParseNode"] = field(default_factory=list)
    production: Production | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def pretty(self) -> str:
        if not self.children:
            return self.symbol
        return f"{self.symbol} → {' '.join(child.pretty() for child in self.children)}"


@dataclass
class ParseResult:
    recognized: bool
    class_name: str | None
    root: ParseNode | None
    steps: list[str]
    ambiguous: bool = False
    message: str = ""


class BottomUpParser:
    """Chart-based recognition with tree reconstruction."""

    def __init__(self, grammar: Grammar = DEFAULT_GRAMMAR) -> None:
        self.grammar = grammar

    def parse(self, text: str) -> dict[str, ParseResult]:
        sequence = [ch for ch in text if not ch.isspace()]
        if not sequence:
            return {
                start: ParseResult(
                    recognized=False,
                    class_name=None,
                    root=None,
                    steps=[],
                    message="Empty sequence.",
                )
                for start in self.grammar.start_symbols
            }

        invalid = [ch for ch in sequence if ch not in self.grammar.terminals]
        if invalid:
            message = "Invalid symbols: " + ", ".join(sorted(set(invalid)))
            return {
                start: ParseResult(
                    recognized=False,
                    class_name=None,
                    root=None,
                    steps=[],
                    message=message,
                )
                for start in self.grammar.start_symbols
            }

        possible = self._build_possible_table(sequence)
        results: dict[str, ParseResult] = {}
        n = len(sequence)

        for start in self.grammar.start_symbols:
            if start not in possible[0][n]:
                results[start] = ParseResult(
                    recognized=False,
                    class_name=None,
                    root=None,
                    steps=[],
                    message="The sequence cannot be derived from " + start,
                )
                continue

            root = self._reconstruct_tree(start, 0, n, sequence, possible, set())
            if root is None:
                results[start] = ParseResult(
                    recognized=False,
                    class_name=None,
                    root=None,
                    steps=[],
                    message="Failed to reconstruct the parse tree.",
                )
                continue

            steps = self._collect_reduction_steps(root)
            results[start] = ParseResult(
                recognized=True,
                class_name=start,
                root=root,
                steps=steps,
                message="",
            )

        recognized_starts = [s for s, result in results.items() if result.recognized]
        if len(recognized_starts) > 1:
            for start in recognized_starts:
                results[start].ambiguous = True

        return results

    def classify(self, text: str) -> ParseResult:
        results = self.parse(text)
        s_res = results["S"]
        t_res = results["T"]

        if s_res.recognized and not t_res.recognized:
            return s_res
        if t_res.recognized and not s_res.recognized:
            return t_res
        if s_res.recognized and t_res.recognized:
            s_depth = self._tree_depth(s_res.root)
            t_depth = self._tree_depth(t_res.root)
            if s_depth < t_depth:
                s_res.ambiguous = True
                return s_res
            if t_depth < s_depth:
                t_res.ambiguous = True
                return t_res
            s_res.ambiguous = True
            return s_res

        return ParseResult(
            recognized=False,
            class_name=None,
            root=None,
            steps=[],
            message="The sequence is not recognized by the grammar.",
        )

    def _build_possible_table(self, sequence: list[str]) -> list[list[set[str]]]:
        n = len(sequence)
        possible: list[list[set[str]]] = [
            [set() for _ in range(n + 1)] for _ in range(n)
        ]

        # Span length 1
        for i, token in enumerate(sequence):
            for production in self.grammar.productions:
                if len(production.rhs) == 1 and production.rhs[0] == token:
                    possible[i][i + 1].add(production.lhs)

        # Span length >= 2
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                for production in self.grammar.productions:
                    if len(production.rhs) != 2:
                        continue
                    left, right = production.rhs
                    for mid in range(i + 1, j):
                        if self._component_matches(possible, sequence, i, mid, left) and self._component_matches(
                            possible, sequence, mid, j, right
                        ):
                            possible[i][j].add(production.lhs)
                            break
        return possible

    def _component_matches(
        self,
        possible: list[list[set[str]]],
        sequence: list[str],
        i: int,
        j: int,
        component: str,
    ) -> bool:
        if component in self.grammar.terminals:
            return j == i + 1 and sequence[i] == component
        return component in possible[i][j]

    def _reconstruct_tree(
        self,
        symbol: str,
        i: int,
        j: int,
        sequence: list[str],
        possible: list[list[set[str]]],
        stack: set[tuple[str, int, int]],
    ) -> ParseNode | None:
        key = (symbol, i, j)
        if key in stack:
            return None
        stack.add(key)

        try:
            # Terminal production
            for production in self.grammar.productions:
                if production.lhs != symbol:
                    continue
                rhs = production.rhs
                if len(rhs) == 1 and rhs[0] in self.grammar.terminals:
                    if j == i + 1 and sequence[i] == rhs[0]:
                        return ParseNode(
                            symbol=symbol,
                            span=(i, j),
                            children=[ParseNode(symbol=rhs[0], span=(i, j), children=[])],
                            production=production,
                        )

            # Binary productions
            for production in self.grammar.productions:
                if production.lhs != symbol or len(production.rhs) != 2:
                    continue
                left, right = production.rhs
                for mid in range(i + 1, j):
                    left_node = self._build_component(left, i, mid, sequence, possible, stack)
                    if left_node is None:
                        continue
                    right_node = self._build_component(right, mid, j, sequence, possible, stack)
                    if right_node is None:
                        continue
                    return ParseNode(
                        symbol=symbol,
                        span=(i, j),
                        children=[left_node, right_node],
                        production=production,
                    )
            return None
        finally:
            stack.remove(key)

    def _build_component(
        self,
        component: str,
        i: int,
        j: int,
        sequence: list[str],
        possible: list[list[set[str]]],
        stack: set[tuple[str, int, int]],
    ) -> ParseNode | None:
        if component in self.grammar.terminals:
            if j == i + 1 and sequence[i] == component:
                return ParseNode(symbol=component, span=(i, j), children=[])
            return None

        if component not in possible[i][j]:
            return None

        return self._reconstruct_tree(component, i, j, sequence, possible, stack)

    def _collect_reduction_steps(self, node: ParseNode) -> list[str]:
        steps: list[str] = []

        def visit(current: ParseNode) -> None:
            for child in current.children:
                visit(child)
            if current.production is not None and current.children:
                rhs_text = " ".join(child.symbol for child in current.children)
                steps.append(f"{rhs_text} → {current.symbol}")

        visit(node)
        return steps

    def _tree_depth(self, node: ParseNode | None) -> int:
        if node is None or not node.children:
            return 1
        return 1 + max(self._tree_depth(child) for child in node.children)


def classify_text(text: str, grammar: Grammar = DEFAULT_GRAMMAR) -> ParseResult:
    return BottomUpParser(grammar).classify(text)
