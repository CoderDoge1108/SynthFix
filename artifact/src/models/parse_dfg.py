"""
SynthFix-Exec: richer symbolic signals via tree-sitter.

Three reference-free / reference-based signals that are strictly stronger
than the bracket-count AST and keyword-LCS CFG currently used:

  * r_parse:  tree-sitter parseability (1 − error-node fraction). Works
              for JS / Python / C and catches syntax failures that a
              chrF-based signal cannot.

  * r_dfg:    def–use edge Jaccard between candidate and target, derived
              from tree-sitter identifier traversal. Approximates a
              lightweight data-flow-graph reward.

  * r_ast_ts: tree-sitter node-type bag-of-ngrams Jaccard between
              candidate and target. A real AST similarity, strictly
              more informative than bracket balance.

All functions are safe: if tree-sitter or the language binding is
missing they fall back to a cheap surface heuristic so the pipeline
never breaks.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple
import re

try:
    # Use tree_sitter_languages for compat with tree_sitter 0.20.x
    # (which the `codebleu` package requires). This loads pre-built
    # grammar .so files via the old Language(path, name) API.
    from tree_sitter_languages import get_parser as _get_parser
    _PARSERS: Dict[str, object] = {}
    for _n in ('python', 'javascript', 'c'):
        _PARSERS[_n] = _get_parser(_n)
    _HAS_TS = True
except Exception as _e:  # pragma: no cover
    _PARSERS = {}
    _HAS_TS = False


def has_treesitter() -> bool:
    return _HAS_TS


def _parse(text: str, lang: str):
    """Return (tree, root_node) or (None, None) on failure."""
    if not _HAS_TS or lang not in _PARSERS:
        return None, None
    try:
        tree = _PARSERS[lang].parse(bytes(text, 'utf8', errors='replace'))
        return tree, tree.root_node
    except Exception:
        return None, None


# ── r_parse ─────────────────────────────────────────────────────────────

def _count_nodes(node, total=[0], errs=[0]):
    total[0] += 1
    if node.type == 'ERROR' or node.is_missing:
        errs[0] += 1
    for ch in node.children:
        _count_nodes(ch, total, errs)


def parse_score(text: str, lang: str) -> float:
    """Return a [0, 1] parseability score via tree-sitter.

    1.0 = parses cleanly with no ERROR / missing nodes.
    Lower = fraction of non-error nodes in the tree.
    Falls back to bracket balance if tree-sitter is unavailable.
    """
    if not text or not text.strip():
        return 0.0
    if not _HAS_TS or lang not in _PARSERS:
        return _bracket_fallback(text)
    tree, root = _parse(text, lang)
    if root is None:
        return _bracket_fallback(text)
    total = [0]; errs = [0]
    _count_nodes(root, total, errs)
    if total[0] == 0:
        return 0.0
    if errs[0] == 0 and not root.has_error:
        return 1.0
    return max(0.0, 1.0 - errs[0] / max(total[0], 1))


def _bracket_fallback(code: str) -> float:
    pairs = [('(', ')'), ('{', '}'), ('[', ']')]
    p = 0
    for o, c in pairs:
        p += abs(code.count(o) - code.count(c))
    return max(0.0, 1.0 - 0.1 * p)


# ── r_dfg: def–use edges via tree-sitter ───────────────────────────────

def _collect_identifiers(node, out: List[Tuple[str, str, int]]):
    """Walk tree, yield (identifier_name, parent_type, depth).

    depth lets us form (name, enclosing-stmt-type) edges as a simple,
    language-agnostic proxy for def–use pairs.
    """
    stack = [(node, 0, 'root')]
    while stack:
        n, d, ptype = stack.pop()
        if n.type == 'identifier' and n.text:
            nm = n.text.decode('utf8', errors='replace')
            out.append((nm, ptype, d))
        for ch in n.children:
            stack.append((ch, d + 1, n.type))


def _dfg_edges(text: str, lang: str) -> Set[Tuple[str, str]]:
    """Return a set of (var_name, enclosing_stmt_type) edges.

    Cheap approximation of def–use: each identifier occurrence pairs
    with the type of its enclosing syntactic construct (e.g.
    ('x', 'assignment'), ('y', 'call_expression')).
    """
    if not text or not _HAS_TS or lang not in _PARSERS:
        return set()
    _, root = _parse(text, lang)
    if root is None:
        return set()
    idents: List[Tuple[str, str, int]] = []
    _collect_identifiers(root, idents)
    edges = set()
    for nm, ptype, _d in idents:
        edges.add((nm, ptype))
    return edges


def dfg_score(generated: str, target: str, lang: str) -> float:
    """Jaccard similarity between (var, enclosing-stmt) edge sets."""
    ge = _dfg_edges(generated, lang)
    te = _dfg_edges(target, lang)
    if not ge and not te:
        return 1.0
    if not ge or not te:
        return 0.0
    inter = len(ge & te)
    union = len(ge | te)
    return inter / union if union else 0.0


# ── r_ast_ts: AST node-type bag similarity ──────────────────────────────

def _node_types(node, out: List[str]):
    stack = [node]
    while stack:
        n = stack.pop()
        out.append(n.type)
        for ch in n.children:
            stack.append(ch)


def ast_ts_score(generated: str, target: str, lang: str) -> float:
    """Jaccard of unigram + bigram of AST node types (tree-sitter).

    Strictly more informative than bracket balance because it actually
    measures structural similarity to the target.
    """
    if not _HAS_TS or lang not in _PARSERS:
        return 0.5
    _, gr = _parse(generated, lang)
    _, tr = _parse(target, lang)
    if gr is None or tr is None:
        return 0.0
    gtypes: List[str] = []; ttypes: List[str] = []
    _node_types(gr, gtypes); _node_types(tr, ttypes)
    gset: Set[str] = set(gtypes)
    tset: Set[str] = set(ttypes)
    if not gset or not tset:
        return 0.0
    u1 = len(gset & tset) / max(len(gset | tset), 1)
    # bigrams (parent→child adjacency) via sequential pairs in traversal
    def bigrams(seq):
        return {f'{seq[i]}>{seq[i+1]}' for i in range(len(seq) - 1)}
    gbg = bigrams(gtypes); tbg = bigrams(ttypes)
    if not gbg or not tbg:
        return u1
    u2 = len(gbg & tbg) / max(len(gbg | tbg), 1)
    return 0.5 * u1 + 0.5 * u2


# ── convenience wrapper ─────────────────────────────────────────────────

def compute_parse_dfg(generated: str, target: str, lang: str) -> Dict[str, float]:
    """One call returns all three tree-sitter-derived scores."""
    return {
        'parse': parse_score(generated, lang),
        'dfg': dfg_score(generated, target, lang),
        'ast_ts': ast_ts_score(generated, target, lang),
    }
