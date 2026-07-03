"""
SynthFix: Router Model

Lightweight MLP that selects SFT or RFT per-sample based on code features.
Trained with REINFORCE policy gradient.

Paper Section 3.2 — Adaptive Training with a Router Model
Features: [f_AST, f_CFG, f_len, f_loss] where f_loss is the model's
current per-sample loss (dynamic signal for adaptive curriculum).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterModel(nn.Module):
    """
    Architecture (Paper Section 3.2, extended with loss feature):
        [f_AST, f_CFG, f_len, f_loss]  ->  64-ReLU  ->  64-ReLU  ->  sigmoid  ->  P(RFT)
    """

    def __init__(self, input_size: int = 4, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def compute_batch_features(code_texts: list,
                           sample_indices: list = None,
                           loss_history: dict = None) -> torch.Tensor:
    """
    Extract features for router (Paper Section 3.2):
        f_B = [f_AST(B), f_CFG(B), f_len(B), f_loss(B)]

    f_loss is each sample's most recent training loss (0.0 if unavailable).
    Returns shape [batch, 4].
    """
    features = []
    for i, text in enumerate(code_texts):
        ast_complexity = text.count('{') + text.count('(') + 1
        cfg_depth = text.count('if') + text.count('for') + text.count('while') + 1
        code_length = len(text.split())
        loss_val = 0.0
        if loss_history is not None and sample_indices is not None:
            idx = sample_indices[i]
            loss_val = loss_history.get(idx, 0.0)
        features.append([ast_complexity, cfg_depth, code_length, loss_val])
    return torch.tensor(features, dtype=torch.float32)


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    Normalize features f_B -> f_tilde_B (Algorithm 1, line 2).
    Min-max normalization to [0, 1] per batch.
    """
    mn = features.min(dim=0, keepdim=True)[0]
    mx = features.max(dim=0, keepdim=True)[0]
    return (features - mn) / (mx - mn + 1e-8)
