"""Build a Python execution-repair benchmark (replaces FixJS).

Train/val: MBPP (974 tasks, real assert test-suites). For each task we
inject a *single* AST mutation into the reference solution and keep it
only if the mutant FAILS >=1 test while the reference passes ALL tests —
i.e. an execution-validated (buggy -> fixed) pair.

Test: QuixBugs (40 real one-line algorithmic bugs) with its own test
suites, so we train on injected bugs and evaluate on *real* bugs.

Output unified JSON ({buggy, fixed, language, tests, ...}) matching the
repo's dataset format, plus a sidecar tests file for the execution oracle.
"""
import argparse
import ast
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
random.seed(42)


# ── Execution harness ────────────────────────────────────────────────
HARNESS = r'''
import sys, json
_PASS = 0; _TOTAL = 0
{setup}
{code}
__TESTS = {tests!r}
for _t in __TESTS:
    _TOTAL += 1
    try:
        exec(_t, globals())
        _PASS += 1
    except Exception:
        pass
print("RESULT", _PASS, _TOTAL)
'''


def run_tests(code, tests, setup='', timeout=10):
    """Return (n_pass, n_total). Subprocess-isolated with a timeout."""
    prog = HARNESS.format(setup=setup or '', code=code, tests=tests)
    try:
        r = subprocess.run([sys.executable, '-c', prog],
                           capture_output=True, text=True, timeout=timeout)
        for line in r.stdout.splitlines()[::-1]:
            if line.startswith('RESULT'):
                _, p, t = line.split()
                return int(p), int(t)
    except Exception:
        pass
    return 0, len(tests)


# ── AST mutation operators ───────────────────────────────────────────
class Mutator(ast.NodeTransformer):
    """Apply exactly one mutation at the `target`-th eligible node."""
    def __init__(self, target):
        self.target = target
        self.count = 0
        self.applied = False

    def _hit(self):
        h = (self.count == self.target)
        self.count += 1
        return h

    def visit_Compare(self, node):
        self.generic_visit(node)
        swaps = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
                 ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
        if node.ops and type(node.ops[0]) in swaps and self._hit():
            node.ops[0] = swaps[type(node.ops[0])]()
            self.applied = True
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        swaps = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv,
                 ast.FloorDiv: ast.Mult}
        if type(node.op) in swaps and self._hit():
            node.op = swaps[type(node.op)]()
            self.applied = True
        return node

    def visit_Constant(self, node):
        self.generic_visit(node)
        if isinstance(node.value, int) and not isinstance(node.value, bool) \
                and self._hit():
            node.value = node.value + random.choice([-1, 1])
            self.applied = True
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self._hit():
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.applied = True
        return node


def inject_bug(code, tests, setup=''):
    """Return a mutant that fails >=1 test, or None."""
    try:
        base = ast.parse(code)
    except SyntaxError:
        return None
    # count eligible nodes
    n = sum(1 for _ in ast.walk(base)
            if isinstance(_, (ast.Compare, ast.BinOp, ast.Constant, ast.BoolOp)))
    order = list(range(n))
    random.shuffle(order)
    for tgt in order[:30]:
        tree = ast.parse(code)
        m = Mutator(tgt)
        new = m.visit(tree)
        if not m.applied:
            continue
        try:
            ast.fix_missing_locations(new)
            mutated = ast.unparse(new)
        except Exception:
            continue
        if mutated.strip() == code.strip():
            continue
        p, t = run_tests(mutated, tests, setup)
        if t > 0 and p < t:                 # fails at least one test
            return mutated
    return None


def build_mbpp(out_records):
    from datasets import load_dataset
    ds = load_dataset('mbpp', 'full')
    items = []
    for split in ['train', 'validation', 'test', 'prompt']:
        items += list(ds[split])
    kept = 0
    for ex in items:
        code = ex['code']
        tests = ex['test_list']
        setup = ex.get('test_setup_code', '') or ''
        # reference must pass ALL tests
        p, t = run_tests(code, tests, setup)
        if t == 0 or p < t:
            continue
        buggy = inject_bug(code, tests, setup)
        if buggy is None:
            continue
        out_records.append({
            'buggy': buggy, 'fixed': code, 'language': 'python',
            'dataset': 'mbpp_repair', 'id': f"mbpp_{ex['task_id']}",
            'tests': tests, 'test_setup': setup,
        })
        kept += 1
        if kept % 50 == 0:
            print(f'[mbpp] kept {kept}', flush=True)
    print(f'[mbpp] total kept {kept} / {len(items)}', flush=True)


def build_quixbugs():
    qb = ROOT / 'data' / 'QuixBugs'
    bp = qb / 'python_programs'
    cp = qb / 'correct_python_programs'
    recs = []
    for f in sorted(bp.glob('*.py')):
        name = f.name
        if name.endswith('_test.py'):
            continue
        corr = cp / name
        if not corr.exists():
            continue
        buggy = f.read_text()
        fixed = corr.read_text()
        # QuixBugs ships pytest files under python_testcases/test_<name>.py;
        # we keep a reference to run them against the candidate function.
        recs.append({
            'buggy': buggy, 'fixed': fixed, 'language': 'python',
            'dataset': 'quixbugs', 'id': f'quixbugs_{name[:-3]}',
            'qb_name': name[:-3],
        })
    print(f'[quixbugs] {len(recs)} programs', flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='data/benchmarks_processed/pyrepair')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mbpp = []
    build_mbpp(mbpp)
    random.shuffle(mbpp)
    n = len(mbpp)
    n_val = max(20, n // 10)
    n_test = max(40, n // 6)
    test = mbpp[:n_test]
    val = mbpp[n_test:n_test + n_val]
    train = mbpp[n_test + n_val:]
    qb = build_quixbugs()

    json.dump(train, open(out / 'train.json', 'w'), indent=1)
    json.dump(val, open(out / 'val.json', 'w'), indent=1)
    # Primary test = MBPP held-out (has assert suites for execution eval).
    json.dump(test, open(out / 'test.json', 'w'), indent=1)
    # Secondary real-bug transfer test = QuixBugs.
    json.dump(qb, open(out / 'test_quixbugs.json', 'w'), indent=1)
    print(f'WROTE train={len(train)} val={len(val)} test(mbpp)={len(test)} '
          f'test_quixbugs={len(qb)} -> {out}', flush=True)


if __name__ == '__main__':
    main()
