"""
SynthFix: Symbolic Reward Model

Composite reward: r(y) = λ_AST * r_AST + λ_CFG * r_CFG + λ_Sem * r_Sem + λ_SIM * r_SIM

Paper Section 3.1 — Symbolic Reward Model for Patch Evaluation
"""

import re


def _try_parse_ast(code: str) -> float:
    """r_AST: continuous syntactic correctness via bracket balance.

    Checks (), {}, [] balance and returns a smooth penalty score
    in [0, 1] rather than discrete bins.
    """
    code = code.strip()
    if not code:
        return 0.0
    pairs = [('(', ')'), ('{', '}'), ('[', ']')]
    penalty = 0
    for o, c in pairs:
        penalty += abs(code.count(o) - code.count(c))
    return max(0.0, 1.0 - 0.1 * penalty)


def _lcs_length(a: list, b: list) -> int:
    """Length of the longest common subsequence."""
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


def _cfg_similarity(generated: str, target: str) -> float:
    """r_CFG: control-flow fidelity via normalized LCS on keyword sequences."""
    cfg_kws = ['if', 'else', 'for', 'while', 'switch', 'case',
               'try', 'catch', 'return', 'break', 'continue', 'throw']
    pattern = r'\b(' + '|'.join(cfg_kws) + r')\b'

    gen_flow = re.findall(pattern, generated)
    tgt_flow = re.findall(pattern, target)
    if not tgt_flow:
        return 1.0 if not gen_flow else 0.8
    if not gen_flow:
        return 0.3
    lcs = _lcs_length(gen_flow, tgt_flow)
    max_len = max(len(gen_flow), len(tgt_flow))
    return lcs / max_len


def _semgrep_heuristic(code: str) -> float:
    """r_Sem: security assessment via vulnerability pattern detection."""
    vuln_patterns = [
        (r'\beval\s*\(', 0.0),
        (r'\bexec\s*\(', 0.0),
        (r'\bsystem\s*\(', 0.1),
        (r'\b__import__\s*\(', 0.1),
        (r'innerHTML\s*=', 0.2),
        (r'document\.write\s*\(', 0.2),
        (r'child_process', 0.1),
        (r'\.exec\s*\(', 0.2),
        (r'sprintf\s*\(', 0.3),
        (r'strcpy\s*\(', 0.2),
        (r'gets\s*\(', 0.1),
    ]
    worst = 1.0
    for pat, penalty_score in vuln_patterns:
        if re.search(pat, code):
            worst = min(worst, penalty_score)
    return worst


def _chrf_similarity(generated: str, target: str, n: int = 6,
                     beta: float = 2.0) -> float:
    """r_SIM: token-level similarity via character n-gram F-score (chrF).

    Lightweight implementation that avoids the sacrebleu import overhead
    while being fully compatible with the standard chrF metric.
    """
    if not generated or not target:
        return 0.0

    def _char_ngrams(text, order):
        ngrams = {}
        for i in range(len(text) - order + 1):
            ng = text[i:i + order]
            ngrams[ng] = ngrams.get(ng, 0) + 1
        return ngrams

    total_precision = 0.0
    total_recall = 0.0
    count = 0
    for order in range(1, n + 1):
        gen_ng = _char_ngrams(generated, order)
        ref_ng = _char_ngrams(target, order)
        if not gen_ng or not ref_ng:
            continue
        overlap = sum(min(gen_ng.get(k, 0), v) for k, v in ref_ng.items())
        gen_total = sum(gen_ng.values())
        ref_total = sum(ref_ng.values())
        p = overlap / gen_total if gen_total else 0.0
        r = overlap / ref_total if ref_total else 0.0
        total_precision += p
        total_recall += r
        count += 1

    if count == 0:
        return 0.0
    avg_p = total_precision / count
    avg_r = total_recall / count
    if avg_p + avg_r == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * avg_p * avg_r / (beta_sq * avg_p + avg_r)


def compute_reward(generated: str, target: str,
                   lambda_ast: float = 0.2,
                   lambda_cfg: float = 0.3,
                   lambda_sem: float = 0.1,
                   lambda_sim: float = 0.4) -> float:
    """
    Composite symbolic reward (Eq. 1 from paper).

    Returns a scalar in [0, 1].
    """
    r_ast = _try_parse_ast(generated)
    r_cfg = _cfg_similarity(generated, target)
    r_sem = _semgrep_heuristic(generated)
    r_sim = _chrf_similarity(generated, target)
    return (lambda_ast * r_ast + lambda_cfg * r_cfg +
            lambda_sem * r_sem + lambda_sim * r_sim)
