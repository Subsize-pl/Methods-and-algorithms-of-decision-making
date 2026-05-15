from collections import defaultdict
from dataclasses import dataclass
from typing import List, Sequence

from .models import (
    ChoiceNode,
    Grammar,
    PatternNode,
    PrefixNode,
    PrefixTrieNode,
    RepeatNode,
    SymbolFactory,
    TerminalNode,
)


def normalize_samples(raw_text: str) -> List[str]:
    """Split user input into unique sample chains."""
    items: List[str] = []
    for line in raw_text.replace(",", "\n").splitlines():
        word = "".join(ch for ch in line.strip() if not ch.isspace())
        if word:
            items.append(word)

    seen = set()
    unique: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def longest_common_prefix(strings: Sequence[str]) -> str:
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        i = 0
        limit = min(len(prefix), len(s))
        while i < limit and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            return ""
    return prefix


def detect_repeat_tail(strings: Sequence[str]) -> tuple[str, str] | None:
    """Detect a pattern like x^k + tail for a set of strings."""
    uniq = sorted(set(strings))
    if len(uniq) < 2:
        return None

    for tail_len in (2, 1):
        if any(len(item) < tail_len for item in uniq):
            continue
        tail = uniq[0][-tail_len:]
        if not all(item.endswith(tail) for item in uniq):
            continue

        prefixes = [item[:-tail_len] for item in uniq]
        if len(set(prefixes)) < 2:
            continue

        nonempty = [prefix for prefix in prefixes if prefix]
        if not nonempty:
            continue

        chars = set()
        ok = True
        for prefix in nonempty:
            if len(set(prefix)) != 1:
                ok = False
                break
            chars.add(prefix[0])

        if ok and len(chars) == 1:
            return next(iter(chars)), tail

    return None


def infer_pattern(strings: Sequence[str]) -> PatternNode:
    """Infer a compact recursive pattern from a set of strings."""
    uniq = sorted(set(strings))
    if not uniq:
        raise ValueError("training sample is empty")

    if len(uniq) == 1:
        return TerminalNode(uniq[0])

    prefix = longest_common_prefix(uniq)
    if prefix:
        stripped = [item[len(prefix) :] for item in uniq]
        return PrefixNode(prefix, infer_pattern(stripped))

    repeat = detect_repeat_tail(uniq)
    if repeat is not None:
        char, tail = repeat
        return RepeatNode(char, TerminalNode(tail))

    groups: dict[str, list[str]] = defaultdict(list)
    for item in uniq:
        groups[item[0]].append(item[1:])

    alts = tuple((ch, infer_pattern(group)) for ch, group in sorted(groups.items()))
    return ChoiceNode(alts)


class ExactGrammarBuilder:
    """Build the exact non-recursive grammar from a sample set."""

    def __init__(self) -> None:
        self.root = PrefixTrieNode()

    def add_sample(self, sample: str) -> None:
        prefix = sample[:-2] if len(sample) > 2 else ""
        tail = tuple(sample[-2:]) if len(sample) > 1 else tuple(sample)

        node = self.root
        for ch in prefix:
            node = node.children.setdefault(ch, PrefixTrieNode())
        node.tails.add(tail)

    def build(self, samples: Sequence[str]) -> Grammar:
        for sample in samples:
            self.add_sample(sample)

        grammar = Grammar(start="S")
        node_to_symbol = {id(self.root): "S"}
        factory = SymbolFactory()
        factory.index = 2  # keep S reserved

        def symbol_for(node: PrefixTrieNode) -> str:
            key = id(node)
            if key not in node_to_symbol:
                node_to_symbol[key] = factory.new()
            return node_to_symbol[key]

        def walk(node: PrefixTrieNode) -> None:
            lhs = symbol_for(node)
            for ch, child in sorted(node.children.items()):
                rhs = (ch, symbol_for(child))
                grammar.add_rule(lhs, rhs)
                walk(child)
            for tail in sorted(node.tails):
                grammar.add_rule(lhs, tail)

        walk(self.root)
        return grammar


class PatternGrammarBuilder:
    """Convert an inferred pattern into a grammar."""

    def __init__(self) -> None:
        self.grammar = Grammar(start="S")
        self.factory = SymbolFactory()
        self.memo: dict[PatternNode, str] = {}

    def build(self, node: PatternNode, *, is_start: bool = False) -> str:
        if node in self.memo:
            return self.memo[node]

        symbol = self.grammar.start if is_start else self.factory.new()
        self.memo[node] = symbol

        if isinstance(node, TerminalNode):
            self.grammar.add_rule(symbol, tuple(node.text))

        elif isinstance(node, RepeatNode):
            self.grammar.add_rule(symbol, (node.char, symbol))
            tail_symbol = self.build(node.tail)
            self.grammar.add_rule(symbol, (tail_symbol,))

        elif isinstance(node, PrefixNode):
            child_symbol = self.build(node.child)
            current = symbol

            if len(node.prefix) == 1:
                self.grammar.add_rule(current, (node.prefix, child_symbol))
            else:
                for ch in node.prefix[:-1]:
                    nxt = self.factory.new()
                    self.grammar.add_rule(current, (ch, nxt))
                    current = nxt
                self.grammar.add_rule(current, (node.prefix[-1], child_symbol))

        elif isinstance(node, ChoiceNode):
            for ch, child in node.alts:
                child_symbol = self.build(child)
                self.grammar.add_rule(symbol, (ch, child_symbol))

        else:
            raise TypeError(f"Unsupported node type: {type(node)!r}")

        return symbol


@dataclass
class SynthesisResult:
    samples: List[str]
    exact_grammar: Grammar
    recursive_grammar: Grammar
    generated_strings: List[str]


class GrammarSynthesizer:
    """Full three-stage synthesis pipeline."""

    def synthesize(
        self,
        samples: Sequence[str],
        *,
        generate_limit: int = 40,
        generate_depth: int = 20,
        generate_max_length: int = 80,
    ) -> SynthesisResult:
        normalized = list(samples)
        exact = ExactGrammarBuilder().build(normalized)
        pattern = infer_pattern(normalized)
        builder = PatternGrammarBuilder()
        builder.build(pattern, is_start=True)
        recursive = builder.grammar.simplify()
        generated = recursive.generate_strings(
            limit=generate_limit,
            max_depth=generate_depth,
            max_length=generate_max_length,
        )
        return SynthesisResult(
            samples=normalized,
            exact_grammar=exact,
            recursive_grammar=recursive,
            generated_strings=generated,
        )
