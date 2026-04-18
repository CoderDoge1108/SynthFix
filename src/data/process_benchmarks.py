"""
Process FixJS, CodeFlaws, and SVEN benchmarks into unified JSON format.

FixJS:      JavaScript — input/{50,50-100,100+}/before_tokenized.txt
CodeFlaws:  C          — <id>-bug-<bugID>-<fixID>/ with two .c files
SVEN:       Python     — data_train_val/{train,val}/cwe-*.jsonl

Output: {split}.json = [{"buggy": ..., "fixed": ..., "language": ...}, ...]
"""

import json
import random
import re
from pathlib import Path


def _save_splits(samples, out, name_tag, ratios=(0.8, 0.1, 0.1)):
    """Shuffle, split, save, print stats."""
    random.seed(13)
    random.shuffle(samples)
    n_train = int(len(samples) * ratios[0])
    n_val = int(len(samples) * ratios[1])
    splits = {
        'train': samples[:n_train],
        'val': samples[n_train:n_train + n_val],
        'test': samples[n_train + n_val:],
    }
    out.mkdir(parents=True, exist_ok=True)
    for sname, data in splits.items():
        (out / f'{sname}.json').write_text(json.dumps(data))
    print(f'[{name_tag}] {len(splits["train"])} train / '
          f'{len(splits["val"])} val / {len(splits["test"])} test',
          flush=True)
    print(f'  Saved to {out}/', flush=True)
    return out


def process_fixjs(raw_dir: str, output_base: Path):
    """Process FixJS (JavaScript) from the raw benchmark format."""
    raw = Path(raw_dir)
    out = output_base / 'fixjs'

    samples = []
    for subdir in ['50', '50-100', '100+']:
        bef = raw / 'input' / subdir / 'before_tokenized.txt'
        aft = raw / 'input' / subdir / 'after_tokenized.txt'
        if not bef.exists():
            continue
        buggy_lines = bef.read_text().splitlines()
        fixed_lines = aft.read_text().splitlines()
        for b, f in zip(buggy_lines, fixed_lines):
            b, f = b.strip(), f.strip()
            if b and f and b != f:
                samples.append({'buggy': b, 'fixed': f,
                                'language': 'javascript'})

    if not samples:
        for fname in ['buggy.txt']:
            buggy_f = raw / 'buggy.txt'
            fixed_f = raw / 'fixed.txt'
            if buggy_f.exists():
                for b, f in zip(buggy_f.read_text().splitlines(),
                                fixed_f.read_text().splitlines()):
                    b, f = b.strip(), f.strip()
                    if b and f and b != f:
                        samples.append({'buggy': b, 'fixed': f,
                                        'language': 'javascript'})

    print(f'[FixJS] Processing JavaScript dataset...', flush=True)
    return _save_splits(samples, out, 'FixJS')


def process_codeflaws(raw_dir: str, output_base: Path):
    """
    Process CodeFlaws (C) dataset.

    Layout: codeflaws/<id>-bug-<bugID>-<fixID>/
      - <id>-<bugID>.c  (buggy)
      - <id>-<fixID>.c  (fixed)
    """
    raw = Path(raw_dir)
    cf_dir = raw / 'codeflaws' if (raw / 'codeflaws').is_dir() else raw
    out = output_base / 'codeflaws'

    samples = []
    bug_dirs = sorted(d for d in cf_dir.iterdir()
                      if d.is_dir() and '-bug-' in d.name)

    for d in bug_dirs:
        m = re.match(r'(.+)-bug-(\d+)-(\d+)', d.name)
        if not m:
            continue
        prefix, bug_id, fix_id = m.group(1), m.group(2), m.group(3)
        c_files = sorted(d.glob('*.c'))
        c_files = [f for f in c_files if not f.name.endswith('.revlog')]
        if len(c_files) < 2:
            continue

        buggy_file = None
        fixed_file = None
        for cf in c_files:
            if bug_id in cf.stem:
                buggy_file = cf
            elif fix_id in cf.stem:
                fixed_file = cf
        if not buggy_file or not fixed_file:
            buggy_file, fixed_file = c_files[0], c_files[1]

        try:
            buggy = buggy_file.read_text(errors='replace').strip()
            fixed = fixed_file.read_text(errors='replace').strip()
        except Exception:
            continue
        if buggy and fixed and buggy != fixed:
            samples.append({'buggy': buggy, 'fixed': fixed, 'language': 'c'})

    print(f'[CodeFlaws] Processing C dataset...', flush=True)
    return _save_splits(samples, out, 'CodeFlaws')


def process_sven(raw_dir: str, output_base: Path):
    """
    Process SVEN (Python) dataset.

    Layout: data_train_val/{train,val}/cwe-*.jsonl
    Each line: {"func_src_before": ..., "func_src_after": ..., "vul_type": ...}
    """
    raw = Path(raw_dir)
    out = output_base / 'sven'

    train_samples = []
    val_samples = []

    for split_name, target in [('train', train_samples), ('val', val_samples)]:
        split_dir = raw / 'data_train_val' / split_name
        if not split_dir.exists():
            continue
        for jf in sorted(split_dir.glob('*.jsonl')):
            for line in jf.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                buggy = obj.get('func_src_before', '').strip()
                fixed = obj.get('func_src_after', '').strip()
                if buggy and fixed and buggy != fixed:
                    target.append({'buggy': buggy, 'fixed': fixed,
                                   'language': 'python'})

    random.seed(13)
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    test_split = int(len(val_samples) * 0.5)
    splits = {
        'train': train_samples,
        'val': val_samples[:test_split],
        'test': val_samples[test_split:],
    }

    out.mkdir(parents=True, exist_ok=True)
    for sname, data in splits.items():
        (out / f'{sname}.json').write_text(json.dumps(data))

    print(f'[SVEN] Processing Python dataset...', flush=True)
    print(f'  SVEN: {len(splits["train"])} train / '
          f'{len(splits["val"])} val / {len(splits["test"])} test',
          flush=True)
    print(f'  Saved to {out}/', flush=True)
    return out
