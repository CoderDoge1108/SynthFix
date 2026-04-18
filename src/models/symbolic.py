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
