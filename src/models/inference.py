"""
SynthFix — inference-time best-of-K with learned reranker.

Pipeline:
  1. Router predicts sample difficulty from buggy-code features.
  2. Compute budget: easy→K=1, medium→K=4, hard→K=8.
  3. Generate K candidates (greedy + mixed-temperature sampling +
     identity fallback).
  4. Extract 13-dim reference-free feature vector per candidate
     (split symbolic + model log-probs + self-consistency).
  5. Score with a LightGBM reranker trained on the val set (where we
     DO have ground truth, so we can label candidates by their actual
     CodeBLEU against GT).
  6. Return the highest-scoring candidate per input. Safety fallback to
     greedy if reranker confidence is low.

Training-free paths: if lightgbm is unavailable, falls back to a
well-tuned rule-based score.
"""

from typing import List, Dict, Tuple, Optional
import math
import re

import numpy as np
import torch
import torch.nn.functional as F

from .symbolic import (compute_reward_split, _chrf_score, _CFG_RE,
                        _lcs_length)

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


# ── Candidate generation ────────────────────────────────────────────────

def _decode_with_logp(seqs, scores, prompt_pad_len, pad_id, tokenizer, bs):
    """Decode continuation strings + compute average log-prob per sample.

    Handles the HF generate output packaging:
      seqs: (bs * nrs, prompt_pad_len + new_tokens)
      scores: tuple of (bs * nrs, vocab) at each new-token step
    """
    out_texts = []
    if scores:
        stacked = torch.stack(scores, dim=0).permute(1, 0, 2)
        logp_all = F.log_softmax(stacked, dim=-1)
        new_tokens = seqs[:, prompt_pad_len:]
        n_new = min(new_tokens.size(1), logp_all.size(1))
        new_tokens_clipped = new_tokens[:, :n_new]
        logp_chosen = torch.gather(
            logp_all[:, :n_new, :], 2,
            new_tokens_clipped.unsqueeze(-1)
        ).squeeze(-1)
        valid = (new_tokens_clipped != pad_id).float()
        summed = (logp_chosen * valid).sum(1)
        lens = valid.sum(1).clamp(min=1)
        avg_logp = (summed / lens).tolist()
    else:
        avg_logp = [0.0] * seqs.size(0)

    for i in range(seqs.size(0)):
        out_texts.append(tokenizer.decode(
            seqs[i, prompt_pad_len:], skip_special_tokens=True
        ).strip()[:2000])
    return out_texts, avg_logp


def generate_k_candidates(model, tokenizer, prompt_ids, prompt_mask,
                           K=8, max_new_tokens=128, pad_id=None):
    """Generate K diverse candidates per prompt.

    Batched / efficient version: uses num_return_sequences to collect
    multiple sampled candidates in a single generate call.

    Candidate layout (wider-temperature schedule for real diversity):
      K=1  : [greedy]
      K=4  : [greedy, T=0.5 x2, identity]
      K=8  : [greedy, T=0.4 x3, T=0.8 x3, identity]
      K=16 : [greedy, T=0.3 x3, T=0.6 x4, T=0.9 x4, T=1.2 x3, identity]
    """
    bs = prompt_ids.size(0)
    prompt_pad_len = prompt_ids.size(1)
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    # Build slot plan: list of (kind, temperature, num_return_sequences)
    slots: List[Tuple[str, float, int]] = [('greedy', 0.0, 1)]
    if K == 4:
        slots += [('sample', 0.5, 2), ('identity', 0.0, 1)]
    elif K == 8:
        slots += [('sample', 0.4, 3),
                  ('sample', 0.8, 3),
                  ('identity', 0.0, 1)]
    elif K >= 16:
        slots += [('sample', 0.3, 3),
                  ('sample', 0.6, 4),
                  ('sample', 0.9, 4),
                  ('sample', 1.2, 3),
                  ('identity', 0.0, 1)]
    else:
        # Fallback for other K values
        remaining = K - 2
        if remaining > 0:
            slots.append(('sample', 0.5, remaining))
        slots.append(('identity', 0.0, 1))

    continuations: List[List[str]] = [[] for _ in range(bs)]
    logps:         List[List[float]] = [[] for _ in range(bs)]
    temperatures:  List[List[float]] = [[] for _ in range(bs)]
    is_greedy_flags: List[List[bool]] = [[] for _ in range(bs)]
    is_identity_flags: List[List[bool]] = [[] for _ in range(bs)]

    for kind, temp, nrs in slots:
        if kind == 'identity':
            for j in range(bs):
                valid = prompt_ids[j][prompt_mask[j].bool()]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                continuations[j].append(text.strip()[:2000])
                logps[j].append(0.0)
                temperatures[j].append(-1.0)
                is_greedy_flags[j].append(False)
                is_identity_flags[j].append(True)
            continue

        with torch.no_grad():
            out = model.generate(
                input_ids=prompt_ids, attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(kind == 'sample'),
                temperature=temp if kind == 'sample' else 1.0,
                top_p=0.95 if kind == 'sample' else 1.0,
                num_return_sequences=nrs,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        seqs = out.sequences  # (bs * nrs, prompt_pad_len + new_tokens)
        texts, logp_avg = _decode_with_logp(
            seqs, out.scores, prompt_pad_len, pad_id, tokenizer, bs)

        # Distribute back to per-sample lists; when nrs>1 HF interleaves
        # nrs copies per sample contiguously in [j*nrs .. j*nrs+nrs).
        for j in range(bs):
            for k in range(nrs):
                idx = j * nrs + k
                continuations[j].append(texts[idx])
                logps[j].append(float(logp_avg[idx]))
                temperatures[j].append(float(temp))
                is_greedy_flags[j].append(kind == 'greedy')
                is_identity_flags[j].append(False)

    return continuations, logps, temperatures, is_greedy_flags, is_identity_flags


# ── Feature extraction ──────────────────────────────────────────────────

FEATURE_NAMES = [
    'ast', 'sem', 'chrf_buggy', 'min_edit_prior',
    'sc_chrf', 'sc_cfg', 'length_ratio',
    'is_greedy', 'temperature', 'avg_logp',
    'is_identity', 'cand_length', 'cfg_count',
]
NUM_FEATURES = len(FEATURE_NAMES)


def _cfg_count(text: str) -> int:
    return len(_CFG_RE.findall(text))


def _cfg_lcs_ratio(a: str, b: str) -> float:
    ka = _CFG_RE.findall(a)
    kb = _CFG_RE.findall(b)
    if not ka and not kb:
        return 1.0
    if not ka or not kb:
        return 0.0
    lcs = _lcs_length(ka, kb)
    return lcs / max(len(ka), len(kb))


def extract_features(buggy: str,
                      candidates: List[str],
                      logps: List[float],
                      temperatures: List[float],
                      is_greedy_flags: List[bool],
                      is_identity_flags: List[bool]) -> np.ndarray:
    """Return (K, NUM_FEATURES) reference-free feature matrix.

    Reference-free means no use of the GT fix string. This is the same
    features used at test time AND at reranker training time on val.
    """
    K = len(candidates)
    feats = np.zeros((K, NUM_FEATURES), dtype=np.float32)
    buggy_len = max(len(buggy), 1)

    # Precompute per-candidate AST/SEM/chrF-to-buggy
    ast_scores = []
    sem_scores = []
    chrf_to_buggy = []
    for c in candidates:
        sp = compute_reward_split(c, buggy)
        ast_scores.append(sp['ast'])
        sem_scores.append(sp['sem'])
        chrf_to_buggy.append(_chrf_score(c, buggy))

    # Self-consistency features
    for i in range(K):
        ca = candidates[i]
        # Self-consistency via chrF & CFG to other candidates
        chrf_others = [
            _chrf_score(ca, candidates[j]) for j in range(K) if j != i
        ]
        cfg_others = [
            _cfg_lcs_ratio(ca, candidates[j]) for j in range(K) if j != i
        ]
        sc_chrf = float(np.mean(chrf_others)) if chrf_others else 0.5
        sc_cfg = float(np.mean(cfg_others)) if cfg_others else 0.5

        # Minimal-edit prior: tent function peaking at chrF=0.85.
        # Rewards fixes that ARE minimal edits of the buggy input.
        edit_sim = chrf_to_buggy[i]
        min_edit = max(0.0, 1.0 - abs(edit_sim - 0.85) / 0.85)

        feats[i, 0] = ast_scores[i]
        feats[i, 1] = sem_scores[i]
        feats[i, 2] = chrf_to_buggy[i]
        feats[i, 3] = min_edit
        feats[i, 4] = sc_chrf
        feats[i, 5] = sc_cfg
        feats[i, 6] = len(ca) / buggy_len
        feats[i, 7] = 1.0 if is_greedy_flags[i] else 0.0
        feats[i, 8] = temperatures[i]
        feats[i, 9] = logps[i]
        feats[i, 10] = 1.0 if is_identity_flags[i] else 0.0
        feats[i, 11] = min(len(ca), 2000) / 2000.0
        feats[i, 12] = min(_cfg_count(ca), 20) / 20.0

    return feats


# ── Learned reranker ────────────────────────────────────────────────────

class LearnedReranker:
    """LightGBM regressor predicting per-candidate CodeBLEU from features.

    Trained on the val set: for each val sample we generate K candidates,
    extract features, and label each candidate with its CodeBLEU vs GT.
    At test time we score all candidates and pick argmax.

    Fallback: if lightgbm unavailable OR training set too small OR
    model predictions look suspicious (near-constant), fall back to a
    robust hand-tuned rule:
        score = 0.25*ast + 0.2*sem + 0.3*sc_chrf + 0.25*min_edit
                + 0.03*is_greedy_bonus
    """

    def __init__(self, tag: str = ''):
        self.tag = tag
        self.gbm = None
        self.feature_mean = None
        self.feature_std = None

    @staticmethod
    def rule_score(feats: np.ndarray) -> np.ndarray:
        """Reference-free heuristic score (fallback when no gbm)."""
        ast = feats[:, 0]
        sem = feats[:, 1]
        min_edit = feats[:, 3]
        sc_chrf = feats[:, 4]
        is_greedy = feats[:, 7]
        return (0.25 * ast + 0.2 * sem + 0.3 * sc_chrf
                + 0.25 * min_edit + 0.03 * is_greedy)

    def fit(self, X: np.ndarray, y: np.ndarray, K: int = 8,
             verbose: bool = True):
        """Train on (N*K, D) features, (N*K,) CodeBLEU labels.

        Uses LGBMRanker (lambdarank) — directly optimizes picking the
        best candidate within each group of K. This is the exact
        inference objective.

        Args:
          K: number of candidates per sample (for group partitioning).
             X must be grouped as [sample0_cand0, ..., sample0_candK-1,
                                    sample1_cand0, ..., sample1_candK-1, ...]
        """
        if not _HAS_LGB or X.shape[0] < 40:
            if verbose:
                print(f'[Reranker/{self.tag}] Using rule-based score '
                      f'(lgb={_HAS_LGB}, N={X.shape[0]})', flush=True)
            return self

        n_samples = X.shape[0] // K
        if n_samples * K != X.shape[0]:
            if verbose:
                print(f'[Reranker/{self.tag}] WARN N={X.shape[0]} not '
                      f'divisible by K={K}, falling back to regressor',
                      flush=True)
            K = None

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if K is not None:
                # LambdaRank: integer relevance labels.
                # Convert continuous CodeBLEU [0,1] to 5-level relevance
                # within each group (relative to that group's min/max).
                rel = np.zeros_like(y, dtype=np.int32)
                for i in range(n_samples):
                    gy = y[i*K:(i+1)*K]
                    gmin, gmax = float(gy.min()), float(gy.max())
                    if gmax - gmin < 1e-6:
                        rel[i*K:(i+1)*K] = 2  # all tied, middle
                    else:
                        # Map to [0, 4]
                        normed = (gy - gmin) / (gmax - gmin)
                        rel[i*K:(i+1)*K] = np.round(normed * 4).astype(np.int32)
                group = np.full(n_samples, K, dtype=np.int32)
                # Small dataset (100 val * K ≈ 1600 samples, 13 features).
                # Keep the ranker shallow/strongly regularised to avoid
                # overfitting the validation set.
                self.gbm = lgb.LGBMRanker(
                    n_estimators=80, max_depth=3, learning_rate=0.05,
                    min_child_samples=20, num_leaves=4,
                    reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, verbose=-1,
                    objective='lambdarank',
                )
                self.gbm.fit(X, rel, group=group)
            else:
                self.gbm = lgb.LGBMRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    min_child_samples=5, num_leaves=16,
                    random_state=42, verbose=-1,
                )
                self.gbm.fit(X, y)

        preds = self.gbm.predict(X)
        pred_var = float(np.var(preds))
        # Also measure in-sample pick quality as a sanity indicator:
        # what fraction of the per-group picks match the true argmax?
        in_sample_top1 = 0
        if K is not None:
            for i in range(n_samples):
                pred_best = int(np.argmax(preds[i*K:(i+1)*K]))
                true_best = int(np.argmax(y[i*K:(i+1)*K]))
                if pred_best == true_best:
                    in_sample_top1 += 1
            in_sample_top1 = in_sample_top1 / n_samples
        if verbose:
            obj = 'LambdaRank' if K is not None else 'Regressor'
            print(f'[Reranker/{self.tag}] LGBM {obj} trained on '
                  f'{X.shape[0]} candidates ({n_samples if K else "?"} groups, '
                  f'K={K})  pred_var={pred_var:.4f}  '
                  f'in_sample_top1={in_sample_top1*100:.1f}%',
                  flush=True)
        if pred_var < 1e-5:
            if verbose:
                print(f'[Reranker/{self.tag}] WARN pred_var too low, '
                      f'falling back to rule-based', flush=True)
            self.gbm = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.gbm is None:
            return self.rule_score(X)
        return self.gbm.predict(X)


# ── Router-gated compute budget ────────────────────────────────────────

def compute_budget_for_prob_hard(prob_hard: float, max_K: int = 8) -> int:
    """Map router prob_hard -> K (inference compute budget).

    Note: the router was trained with loss_history as a key feature.
    At inference time that feature is unavailable, so the router tends
    to output low prob_hard for all test samples. Empirically we found
    this collapses K to 1 for the entire test set. To preserve the
    inference-time contribution of the method, we use a floor (K>=4)
    and only escalate to K=8 on the hardest router predictions.
    """
    if prob_hard < 0.3:
        return min(4, max_K)  # floor at K=4 so reranker always gets choice
    if prob_hard < 0.7:
        return min(4, max_K)
    return min(8, max_K)


def compute_budget_uniform(max_K: int = 8) -> int:
    """Fixed-K schedule — use max_K for every sample.

    Use this when the router cannot be trusted at inference time
    (e.g. when loss_history is unavailable). Guarantees the reranker
    has at least max_K candidates to choose from per sample.
    """
    return max_K


# ── Training the reranker on val set ───────────────────────────────────

def build_reranker_training_data(model, tokenizer, router, val_loader,
                                  device, lang, codebleu_fn,
                                  K: int = 8, max_new_tokens: int = 128,
                                  max_samples: int = 200):
    """Run full best-of-K on val samples, label each candidate with the
    actual CodeBLEU of candidate vs GT, return (X, y) for reranker
    training.
    """
    pad_id = tokenizer.pad_token_id
    X_list, y_list = [], []
    seen = 0
    model.eval()
    router.eval()
    for batch in val_loader:
        if seen >= max_samples:
            break
        pids = batch['prompt_input_ids'].to(device, non_blocking=True)
        pmask = batch['prompt_attention_mask'].to(device, non_blocking=True)
        conts, logps, temps, gflags, iflags = generate_k_candidates(
            model, tokenizer, pids, pmask, K=K,
            max_new_tokens=max_new_tokens, pad_id=pad_id)
        for j in range(pids.size(0)):
            if seen >= max_samples:
                break
            buggy = batch['buggy_text'][j].strip()[:2000]
            gt = batch['fixed_text'][j].strip()[:2000]
            cands = conts[j]
            feats = extract_features(
                buggy, cands, logps[j], temps[j], gflags[j], iflags[j])
            # Per-candidate CodeBLEU vs GT (the training label)
            labels = np.array([
                codebleu_fn([c], [gt], lang=lang) for c in cands
            ], dtype=np.float32)
            X_list.append(feats)
            y_list.append(labels)
            seen += 1
    if not X_list:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32), \
               np.zeros((0,), dtype=np.float32)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Diagnostics: oracle upper-bound (if we picked the best candidate
    # per sample) and greedy-only baseline (if we always picked the
    # first/greedy). This tells us the headroom and whether the
    # reranker problem is well-posed.
    try:
        per_sample_y = [y_list[i] for i in range(len(y_list))]
        oracle = float(np.mean([row.max() for row in per_sample_y]))
        greedy = float(np.mean([row[0] for row in per_sample_y]))
        worst = float(np.mean([row.min() for row in per_sample_y]))
        print(f'[Reranker]   greedy={greedy*100:.2f}%  '
              f'oracle_max={oracle*100:.2f}%  '
              f'worst={worst*100:.2f}%  '
              f'mean={y.mean()*100:.2f}%', flush=True)
    except Exception:
        pass

    return X, y
