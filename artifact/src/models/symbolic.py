"""
SynthFix v10: Split symbolic feature extraction.

This module is the shared symbolic layer used at both training time
(per-token loss weighting) and inference time (per-candidate reranking).

Design:
  * Every target token is mapped to exactly one of four symbolic types
    {AST, CFG, SEM, SIM}. The mapping is language-agnostic and operates
    purely on decoded token strings, so it works for JS / Python / C.
  * compute_reward_split returns each symbolic component in [0, 1]
    separately, enabling fine-grained supervision and reranking.

The four components:
  * AST  — structural syntactic correctness (bracket balance)
  * CFG  — control-flow fidelity vs. reference (LCS on keywords)
  * SEM  — security heuristic (vulnerability pattern detection)
  * SIM  — surface similarity (chrF) vs. reference
"""

import re
from typing import List, Dict

# ── Token classification ────────────────────────────────────────────────

_AST_PATTERN = re.compile(r'^[\s]*[(){}\[\];,]+[\s]*$')
_CFG_KEYWORDS = {
    'if', 'else', 'elif', 'for', 'while', 'switch', 'case', 'default',
    'try', 'catch', 'except', 'finally', 'return', 'break', 'continue',
    'throw', 'raise', 'yield', 'do',
}
_SEM_PATTERNS = [
    re.compile(r'\beval\b'), re.compile(r'\bexec\b'),
    re.compile(r'\bsystem\b'), re.compile(r'\b__import__\b'),
    re.compile(r'innerHTML'), re.compile(r'document\.write'),
    re.compile(r'child_process'), re.compile(r'sprintf'),
    re.compile(r'\bstrcpy\b'), re.compile(r'\bgets\b'),
    re.compile(r'\bshell=True\b'), re.compile(r'\bunsafe\b'),
    re.compile(r'\bsanitize'), re.compile(r'\bescape'),
    re.compile(r'\bvalidate'), re.compile(r'\bpickle\b'),
    re.compile(r'\bsubprocess'), re.compile(r'assertEqual'),
    re.compile(r'\bre\.(?:escape|compile)'),
]


def classify_token_string(tok: str) -> int:
    """Return 0=AST, 1=CFG, 2=SEM, 3=SIM for a single token string.

    Token strings may include a leading space (BPE convention) or be
    empty. We strip and lowercase-check keywords while preserving the
    raw string for regex-based SEM detection.
    """
    if not tok:
        return 3  # SIM default
    stripped = tok.strip()
    if not stripped:
        return 3
    if _AST_PATTERN.match(tok):
        return 0
    lower = stripped.lower()
    if lower in _CFG_KEYWORDS:
        return 1
    for pat in _SEM_PATTERNS:
        if pat.search(tok):
            return 2
    return 3


def classify_token_ids(token_ids: List[int], tokenizer) -> List[int]:
    """Classify each token id into one of {AST, CFG, SEM, SIM}.

    Decodes each id individually so BPE subword structure is preserved
    (tokens like ' if', '{' keep their form).
    """
    tags = []
    for tid in token_ids:
        try:
            s = tokenizer.decode([int(tid)], skip_special_tokens=False)
        except Exception:
            s = ''
        tags.append(classify_token_string(s))
    return tags


# ── Split symbolic reward ──────────────────────────────────────────────

def _ast_score(code: str) -> float:
    code = code.strip()
    if not code:
        return 0.0
    pairs = [('(', ')'), ('{', '}'), ('[', ']')]
    penalty = 0
    for o, c in pairs:
        penalty += abs(code.count(o) - code.count(c))
    return max(0.0, 1.0 - 0.1 * penalty)


def _lcs_length(a: list, b: list) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    if m > 500 or n > 500:
        a, b = a[:500], b[:500]
        m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


_CFG_RE = re.compile(
    r'\b(if|else|for|while|switch|case|try|catch|return|'
    r'break|continue|throw|except|raise|yield)\b')


def _cfg_score(generated: str, target: str) -> float:
    gen_flow = _CFG_RE.findall(generated)
    tgt_flow = _CFG_RE.findall(target)
    if not tgt_flow:
        return 1.0 if not gen_flow else 0.8
    if not gen_flow:
        return 0.3
    lcs = _lcs_length(gen_flow, tgt_flow)
    max_len = max(len(gen_flow), len(tgt_flow))
    return lcs / max_len


_VULN_PATTERNS = [
    (re.compile(r'\beval\s*\('), 0.0),
    (re.compile(r'\bexec\s*\('), 0.0),
    (re.compile(r'\bsystem\s*\('), 0.1),
    (re.compile(r'\b__import__\s*\('), 0.1),
    (re.compile(r'innerHTML\s*='), 0.2),
    (re.compile(r'document\.write\s*\('), 0.2),
    (re.compile(r'child_process'), 0.1),
    (re.compile(r'\.exec\s*\('), 0.2),
    (re.compile(r'sprintf\s*\('), 0.3),
    (re.compile(r'strcpy\s*\('), 0.2),
    (re.compile(r'gets\s*\('), 0.1),
    (re.compile(r'shell\s*=\s*True'), 0.1),
    (re.compile(r'pickle\.loads\s*\('), 0.2),
]


def _sem_score(code: str) -> float:
    worst = 1.0
    for pat, penalty in _VULN_PATTERNS:
        if pat.search(code):
            worst = min(worst, penalty)
    return worst


def _chrf_score(generated: str, target: str, n: int = 6,
                beta: float = 2.0) -> float:
    if not generated or not target:
        return 0.0

    def _char_ngrams(text, order):
        d = {}
        for i in range(len(text) - order + 1):
            ng = text[i:i + order]
            d[ng] = d.get(ng, 0) + 1
        return d

    total_p = 0.0
    total_r = 0.0
    count = 0
    for order in range(1, n + 1):
        g = _char_ngrams(generated, order)
        r = _char_ngrams(target, order)
        if not g or not r:
            continue
        overlap = sum(min(g.get(k, 0), v) for k, v in r.items())
        gt = sum(g.values())
        rt = sum(r.values())
        p = overlap / gt if gt else 0.0
        rr = overlap / rt if rt else 0.0
        total_p += p
        total_r += rr
        count += 1
    if count == 0:
        return 0.0
    avg_p = total_p / count
    avg_r = total_r / count
    if avg_p + avg_r == 0:
        return 0.0
    bs = beta ** 2
    return (1 + bs) * avg_p * avg_r / (bs * avg_p + avg_r)


def compute_reward_split(generated: str, target: str) -> Dict[str, float]:
    """Return the 4 symbolic reward components separately.

    {'ast', 'cfg', 'sem', 'sim'} — each in [0, 1].
    """
    return {
        'ast': _ast_score(generated),
        'cfg': _cfg_score(generated, target),
        'sem': _sem_score(generated),
        'sim': _chrf_score(generated, target),
    }


def compute_reward_from_split(split: Dict[str, float],
                              lambda_ast: float = 0.2,
                              lambda_cfg: float = 0.3,
                              lambda_sem: float = 0.1,
                              lambda_sim: float = 0.4) -> float:
    """Collapse a split reward back to a scalar (matches old compute_reward)."""
    return (lambda_ast * split['ast'] + lambda_cfg * split['cfg']
            + lambda_sem * split['sem'] + lambda_sim * split['sim'])


# ── Rich reward ────────────────────────────────────────────────────────
# Adds three semantic-level signals that complement surface chrF-to-GT.
# These signals provide symbolic evidence while retaining the surface
# similarity anchor used by the base reward.
#
#   r_parse : tree-sitter parseability (1 - error-node fraction)
#   r_dfg   : def-use dependency Jaccard against the *target* fix
#   r_lint  : language-aware static-analyzer cleanliness
#                 (pyflakes for Python, gcc -Wall for C,
#                  tree-sitter undeclared-ref heuristic for JS)

def compute_reward_split_rich(generated: str, target: str,
                               lang: str = 'javascript') -> Dict[str, float]:
    """Return the 4 base + 3 rich reward components separately.

    Returns a dict with keys
        {ast, cfg, sem, sim, parse, dfg, lint}, each in [0, 1].

    The rich keys are filled with 0.5 (neutral) when the respective
    optional module is unavailable, so training cost is predictable.
    """
    base = compute_reward_split(generated, target)
    try:
        from .parse_dfg import parse_score, dfg_score
        base['parse'] = float(parse_score(generated, lang))
        base['dfg'] = float(dfg_score(generated, target, lang))
    except Exception:
        base['parse'] = 0.5
        base['dfg'] = 0.5
    try:
        from .lint_reward import lint_score
        base['lint'] = float(lint_score(generated, lang))
    except Exception:
        base['lint'] = 0.5
    return base


def compute_reward_from_split_rich(split: Dict[str, float],
                                    lambda_ast: float = 0.05,
                                    lambda_cfg: float = 0.15,
                                    lambda_sem: float = 0.05,
                                    lambda_sim: float = 0.30,
                                    lambda_parse: float = 0.15,
                                    lambda_dfg: float = 0.15,
                                    lambda_lint: float = 0.15,
                                    lambda_exec: float = 0.0,
                                    lambda_repair_effect: float = 0.0) -> float:
    """Collapse a rich split reward. Default weights give 45 % to the
    three semantic signals and 55 % to the base four.
    Set ``lambda_exec > 0`` when ``split`` contains an ``exec`` key
    from Codeflaws-style test-case execution."""
    return (lambda_ast * split.get('ast', 0.5)
            + lambda_cfg * split.get('cfg', 0.5)
            + lambda_sem * split.get('sem', 0.5)
            + lambda_sim * split.get('sim', 0.5)
            + lambda_parse * split.get('parse', 0.5)
            + lambda_dfg * split.get('dfg', 0.5)
            + lambda_lint * split.get('lint', 0.5)
            + lambda_exec * split.get('exec', 0.5)
            + lambda_repair_effect * split.get('repair_effect', 0.5))


def compute_reward_split_rich_exec(generated: str, target: str,
                                    lang: str = 'c',
                                    test_dir: str = '',
                                    buggy: str = '',
                                    max_tests: int = 5) -> Dict[str, float]:
    """Rich reward *plus* compiled-and-executed test-case pass rate.

    Only meaningful for Codeflaws-style items where ``test_dir`` points
    to a directory with ``input-pos*`` / ``output-pos*`` files. When
    unavailable (empty ``test_dir`` or missing tests) the ``exec`` key
    is filled with 0.5 (neutral) so training cost stays bounded and
    the advantage signal degenerates gracefully to the static rich
    reward on those items.
    """
    base = compute_reward_split_rich(generated, target, lang=lang)
    try:
        from .repair_effect import repair_effect_score
        base['repair_effect'] = float(
            repair_effect_score(generated, buggy, lang=lang))
    except Exception:
        base['repair_effect'] = 0.5
    if not test_dir:
        base['exec'] = 0.5
        return base
    try:
        from pathlib import Path as _P
        from .exec_reward import exec_reward as _exec_reward
        res = _exec_reward(generated, _P(test_dir), max_tests=max_tests)
        # 0.5 when no tests were discovered (can't reward functional
        # correctness) so the signal is neutral, not punitive.
        if res.total == 0:
            base['exec'] = 0.5
        else:
            base['exec'] = float(res.pass_rate)
    except Exception:
        base['exec'] = 0.5
    return base
