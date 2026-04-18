"""
SynthFix: Dataset Utilities

PyTorch Dataset and DataLoader for vulnerability repair with buggy/fixed code pairs.

Paper Reference: Section 4 — Experimental Design
                 Datasets: FixJS (JavaScript, ~300k), CodeFlaws (C, ~4k)
                 Split: 80/10/10 with seed=13
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class RepairDataset(Dataset):
    """
    Dataset for code vulnerability repair.

    Loads buggy/fixed code pairs from unified JSON format:
        [{"buggy": str, "fixed": str, "language": str, ...}, ...]

    Also supports legacy text format (split_buggy.txt / split_fixed.txt).

    Args:
        data_dir:   Path to dataset directory
        split:      'train', 'val', or 'test'
        max_length: Maximum token sequence length (default: 512)
    """

    def __init__(self, data_dir, split='train', max_length=512):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length

        # Try JSON format first (unified format)
        json_file = self.data_dir / f"{split}.json"
        if json_file.exists():
            data = json.loads(json_file.read_text())
            self.buggy_codes = [item['buggy'] for item in data]
            self.fixed_codes = [item['fixed'] for item in data]
        else:
            # Fall back to legacy text format
            buggy_file = self.data_dir / f"{split}_buggy.txt"
            fixed_file = self.data_dir / f"{split}_fixed.txt"
            if buggy_file.exists() and fixed_file.exists():
                self.buggy_codes = buggy_file.read_text().splitlines()
                self.fixed_codes = fixed_file.read_text().splitlines()
            else:
                raise FileNotFoundError(
                    f"No data found in {self.data_dir} for split '{split}'. "
                    f"Expected {json_file} or {buggy_file} + {fixed_file}."
                )

        assert len(self.buggy_codes) == len(self.fixed_codes), \
            f"Mismatch: {len(self.buggy_codes)} buggy vs {len(self.fixed_codes)} fixed"

    def __len__(self):
        return len(self.buggy_codes)

    def __getitem__(self, idx):
        return {
            'buggy': self.buggy_codes[idx],
            'fixed': self.fixed_codes[idx],
            'index': idx,
        }


SEP = "\n<FIX>\n"


def collate_fn(batch, tokenizer, max_length=512):
    """Causal-LM collate for seq2seq repair: input = [buggy, fixed],
    labels = [-100 * len(buggy), fixed_tokens, -100 * pad].

    This is the standard Hugging Face pattern for instruction tuning /
    causal-LM supervised fine-tuning: the model is trained to predict
    `fixed` tokens conditioned on the preceding `buggy` prompt. Only the
    target portion contributes to the cross-entropy loss; prompt and
    padding tokens are masked with -100.

    `buggy_text` and `fixed_text` are kept as raw strings for the router
    (feature extraction) and reward (symbolic scoring).

    Also returns `prompt_lens` so downstream code knows where the prompt
    ends in each input.
    """
    buggy = [item['buggy'] for item in batch]
    fixed = [item['fixed'] for item in batch]
    indices = [item['index'] for item in batch]

    # 1. Tokenize prompts alone to know where target starts per sample.
    #    Reserve half the budget for prompt so target fits.
    prompt_budget = max(64, max_length // 2)
    target_budget = max_length - prompt_budget

    prompt_tok = tokenizer(
        buggy,
        max_length=prompt_budget,
        truncation=True,
        add_special_tokens=True,
        padding=False,
    )
    target_tok = tokenizer(
        fixed,
        max_length=target_budget,
        truncation=True,
        add_special_tokens=False,
        padding=False,
    )

    pad_id = tokenizer.pad_token_id
    bs = len(batch)

    input_ids = torch.full((bs, max_length), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((bs, max_length), dtype=torch.long)
    labels = torch.full((bs, max_length), -100, dtype=torch.long)
    prompt_lens = []

    # Prompt-only tensors (for inference: model.generate should see
    # only the buggy prompt, not the target). Left-padded so that the
    # prompt ends flush with the right edge, which is what HF generate
    # expects for batched causal generation.
    max_prompt_len = max(len(p) for p in prompt_tok['input_ids'])
    prompt_input_ids = torch.full((bs, max_prompt_len), pad_id, dtype=torch.long)
    prompt_attn_mask = torch.zeros((bs, max_prompt_len), dtype=torch.long)

    for i in range(bs):
        p_ids = prompt_tok['input_ids'][i]
        t_ids = target_tok['input_ids'][i]
        if tokenizer.eos_token_id is not None:
            t_ids = list(t_ids) + [tokenizer.eos_token_id]
        full_ids = list(p_ids) + list(t_ids)
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            t_ids = t_ids[:max_length - len(p_ids)]
        n = len(full_ids)
        input_ids[i, :n] = torch.tensor(full_ids, dtype=torch.long)
        attn_mask[i, :n] = 1
        tgt_start = len(p_ids)
        tgt_end = tgt_start + len(t_ids)
        tgt_end = min(tgt_end, max_length)
        if tgt_start < tgt_end:
            labels[i, tgt_start:tgt_end] = torch.tensor(
                full_ids[tgt_start:tgt_end], dtype=torch.long)
        prompt_lens.append(tgt_start)

        # Left-pad the prompt for generation
        pl = len(p_ids)
        prompt_input_ids[i, max_prompt_len - pl:] = torch.tensor(
            p_ids, dtype=torch.long)
        prompt_attn_mask[i, max_prompt_len - pl:] = 1

    return {
        'input_ids': input_ids,
        'attention_mask': attn_mask,
        'labels': labels,
        'prompt_input_ids': prompt_input_ids,
        'prompt_attention_mask': prompt_attn_mask,
        'buggy_text': buggy,
        'fixed_text': fixed,
        'sample_indices': indices,
        'prompt_lens': prompt_lens,
    }


def create_dataloaders(data_dir, tokenizer, batch_size=16, max_length=512,
                       num_workers=4, shuffle_seed=42):
    """
    Create train/val/test DataLoaders from a processed dataset directory.

    Args:
        data_dir:      Path to directory with {train,val,test}.json
        tokenizer:     HuggingFace tokenizer
        batch_size:    Batch size
        max_length:    Maximum token sequence length
        num_workers:   DataLoader worker processes (0 = main thread)
        shuffle_seed:  Seed for the training-loader shuffle generator.
                       This ensures deterministic, method-independent
                       batch ordering (SFT and SynthFix see the same
                       sequence of batches regardless of prior RNG
                       consumption by model/router initialization).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = RepairDataset(data_dir, split='train', max_length=max_length)
    val_ds = RepairDataset(data_dir, split='val', max_length=max_length)
    test_ds = RepairDataset(data_dir, split='test', max_length=max_length)

    fn = lambda b: collate_fn(b, tokenizer, max_length)

    use_persistent = num_workers > 0

    # Dedicated generator so shuffle order is reproducible across methods.
    train_gen = torch.Generator()
    train_gen.manual_seed(shuffle_seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=fn,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=fn,
        num_workers=min(num_workers, 2), pin_memory=True,
        persistent_workers=use_persistent and num_workers >= 2,
        prefetch_factor=2 if (use_persistent and num_workers >= 2) else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=fn,
        num_workers=min(num_workers, 2), pin_memory=True,
        persistent_workers=use_persistent and num_workers >= 2,
        prefetch_factor=2 if (use_persistent and num_workers >= 2) else None,
    )

    print(f"Dataset loaded from {data_dir}:")
    print(f"  Train: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_ds)} samples ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_ds)} samples ({len(test_loader)} batches)")
    print(f"  Workers: {num_workers} | pin_memory: True")

    return train_loader, val_loader, test_loader
