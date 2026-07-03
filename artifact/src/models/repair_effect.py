"""Reference-free local repair-effect reward.

This is a lightweight implementation of the patch-effect signal used by
SynthFix: score whether a candidate moves away from the buggy program in a
repair-oriented direction, without using the fixed reference.

It is intentionally conservative and lexical. Strong preferences are only
given when the candidate adds evidence that is common in security repairs:
bounds/null/divisor guards, early error paths, cleanup/reset, or safer API
use. Ambiguous edits get small or neutral scores; OLD-like candidates are
penalized. The signal is meant to complement execution tests and chrF, not
replace them.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Set


_IDENT = r'[A-Za-z_][A-Za-z0-9_]*'
_CMP_RE = re.compile(
    rf'(?P<lhs>{_IDENT}(?:\s*(?:->|\.|\[)\s*{_IDENT}?\s*\]?)*|\d+)'
    r'\s*(?P<op><=|>=|<|>|==|!=)\s*'
    rf'(?P<rhs>{_IDENT}|\d+|sizeof\s*\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*_MAX)',
)
_IF_RE = re.compile(r'\b(?:if|while)\s*\((?P<cond>[^)]{1,240})\)')
_RET_RE = re.compile(
    r'\b(return|goto|break|continue|exit\s*\(|abort\s*\(|'
    r'fprintf\s*\(|perror\s*\()'
)
_NULL_RE = re.compile(r'\b(NULL|nullptr|0)\b')
_ALLOC_RE = re.compile(r'\b(malloc|calloc|realloc|new)\s*\(')
_FREE_RE = re.compile(r'\b(free|delete|release|close)\s*\(')
_SAFE_API_RE = re.compile(
    r'\b(snprintf|strncpy|strncat|memcpy_s|strlcpy|strlcat|'
    r'calloc|fgets)\s*\('
)
_UNSAFE_API_RE = re.compile(r'\b(sprintf|strcpy|strcat|gets)\s*\(')
_ARRAY_ACCESS_RE = re.compile(rf'(?P<arr>{_IDENT})\s*\[\s*(?P<idx>{_IDENT})\s*\]')
_DIV_RE = re.compile(r'[/%]\s*(?P<den>[A-Za-z_][A-Za-z0-9_]*)')


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip())


def _set_added(old: Iterable[str], new: Iterable[str]) -> Set[str]:
    return set(new) - set(old)


def _comparisons(code: str) -> Set[str]:
    out = set()
    for m in _CMP_RE.finditer(code or ''):
        lhs = _norm(m.group('lhs'))
        rhs = _norm(m.group('rhs'))
        op = m.group('op')
        out.add(f'{lhs} {op} {rhs}')
    return out


def _guards(code: str) -> Set[str]:
    return {_norm(m.group('cond')) for m in _IF_RE.finditer(code or '')}


def _error_guards(code: str) -> Set[str]:
    """Return guards whose nearby body has an error/exit path."""
    out = set()
    text = code or ''
    for m in _IF_RE.finditer(text):
        cond = _norm(m.group('cond'))
        window = text[m.end():m.end() + 220]
        if _RET_RE.search(window):
            out.add(cond)
    return out


def _array_indices(code: str) -> Set[str]:
    return {m.group('idx') for m in _ARRAY_ACCESS_RE.finditer(code or '')}


def _div_denominators(code: str) -> Set[str]:
    return {m.group('den') for m in _DIV_RE.finditer(code or '')}


def _mentions_any(expr: str, names: Set[str]) -> bool:
    if not names:
        return False
    toks = set(re.findall(_IDENT, expr or ''))
    return bool(toks & names)


def repair_effect_split(candidate: str, buggy: str, lang: str = 'c'
                        ) -> Dict[str, float]:
    """Return reference-free repair-effect components in [0, 1].

    Keys:
      guard_delta   : adds guard conditions relative to buggy
      bounds_delta  : adds comparisons mentioning array indices
      error_path    : added guard locally leads to return/goto/error
      arithmetic    : adds divisor guards
      lifecycle     : cleanup/reset/free evidence
      api_safety    : safer API use or removal of unsafe API
      old_like      : inverse signal; 0 means candidate is very OLD-like
    """
    cand = candidate or ''
    old = buggy or ''
    old_guards = _guards(old)
    cand_guards = _guards(cand)
    added_guards = _set_added(old_guards, cand_guards)

    old_cmps = _comparisons(old)
    cand_cmps = _comparisons(cand)
    added_cmps = _set_added(old_cmps, cand_cmps)
    idx_names = _array_indices(old) | _array_indices(cand)

    old_err = _error_guards(old)
    cand_err = _error_guards(cand)
    added_err = _set_added(old_err, cand_err)

    denoms = _div_denominators(old) | _div_denominators(cand)
    divisor_guard = any(_mentions_any(g, denoms) for g in added_guards)
    bounds_guard = any(_mentions_any(c, idx_names) for c in added_cmps)

    # Lifecycle signal: cleanup calls or reset-to-null added near candidate.
    old_life = len(_FREE_RE.findall(old)) + len(re.findall(r'=\s*NULL\b', old))
    cand_life = len(_FREE_RE.findall(cand)) + len(re.findall(r'=\s*NULL\b', cand))

    old_safe = len(_SAFE_API_RE.findall(old))
    cand_safe = len(_SAFE_API_RE.findall(cand))
    old_unsafe = len(_UNSAFE_API_RE.findall(old))
    cand_unsafe = len(_UNSAFE_API_RE.findall(cand))

    # OLD-like: high chr overlap with no new repair evidence is risky.
    try:
        from .symbolic import _chrf_score
        old_sim = float(_chrf_score(cand, old))
    except Exception:
        old_sim = 1.0 if _norm(cand) == _norm(old) else 0.5

    any_strong = bool(added_err or bounds_guard or divisor_guard
                      or cand_safe > old_safe or cand_unsafe < old_unsafe
                      or cand_life > old_life)
    old_like = 0.0 if old_sim > 0.985 and not any_strong else max(0.0, 1.0 - old_sim)

    return {
        'guard_delta': min(1.0, len(added_guards) / 3.0),
        'bounds_delta': 1.0 if bounds_guard else 0.5 if added_cmps else 0.0,
        'error_path': min(1.0, len(added_err) / 2.0),
        'arithmetic': 1.0 if divisor_guard else 0.0,
        'lifecycle': 1.0 if cand_life > old_life else 0.0,
        'api_safety': 1.0 if (cand_safe > old_safe or cand_unsafe < old_unsafe)
        else 0.0,
        'old_like': old_like,
    }


def repair_effect_score(candidate: str, buggy: str, lang: str = 'c') -> float:
    """Collapse local repair-effect evidence into a scalar [0, 1]."""
    s = repair_effect_split(candidate, buggy, lang=lang)
    raw = (
        0.16 * s['guard_delta']
        + 0.22 * s['bounds_delta']
        + 0.22 * s['error_path']
        + 0.12 * s['arithmetic']
        + 0.10 * s['lifecycle']
        + 0.10 * s['api_safety']
        + 0.08 * s['old_like']
    )
    return max(0.0, min(1.0, raw))

