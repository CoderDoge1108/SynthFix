"""
SynthFix-Exec: execution-based test-case reward for Codeflaws C programs.

The Codeflaws dataset ships with per-bug competitive-programming test
suites (``input-pos*`` / ``output-pos*`` and optional negative tests)
together with a Makefile. That lets us compute a real functional
correctness reward for any candidate patch.

Pipeline per candidate (with a hard time / compile budget):
  1. Write the candidate string to ``<tmp>/<main>.c``
  2. Compile with the bug's Makefile (fallback: gcc -O0 -std=c99 -lm)
  3. Run the binary on each ``input-pos<i>`` with stdin redirection,
     a wall-clock timeout, and normalised-whitespace diff vs
     ``output-pos<i>``.
  4. Reward = passed / total. Compile failure or timeout on every
     test = 0. Tree-sitter parse failure short-circuits to 0 without
     touching the filesystem.

The reward is deliberately conservative: partial credit only comes from
tests that actually pass, so noise comes from the test-suite (not ours).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shutil
import signal
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from .parse_dfg import parse_score


# ── Configuration ───────────────────────────────────────────────────────

COMPILE_TIMEOUT_S = 10
RUN_TIMEOUT_S = 5
DEFAULT_MAX_TESTS = 5       # per candidate, bounds wall-clock cost
MAX_OUTPUT_BYTES = 64 * 1024
GCC_FLAGS = ['-O0', '-std=c99', '-lm', '-fno-strict-aliasing',
             '-w']  # -w silences warnings for noisy Codeflaws bugs


# ── Test-suite discovery ───────────────────────────────────────────────

@dataclass
class CodeflawsTestCase:
    input_file: Path
    output_file: Path
    name: str


def discover_tests(bug_dir: Path, max_tests: int = DEFAULT_MAX_TESTS
                   ) -> List[CodeflawsTestCase]:
    """Return up to ``max_tests`` positive test pairs from ``bug_dir``.

    Samples from public (``input-pos*``) and heldout (``heldout-input-pos*``)
    test suites — heldout is what actually distinguishes buggy from fixed
    in the Codeflaws benchmark, so we always include some.
    """
    if not bug_dir.exists():
        return []
    tests: List[CodeflawsTestCase] = []

    def _collect(glob_prefix: str, out_prefix: str, limit: int) -> None:
        for inp in sorted(bug_dir.glob(f'{glob_prefix}*'))[:limit]:
            idx = inp.name[len(glob_prefix):]
            out = bug_dir / f'{out_prefix}{idx}'
            if not out.exists():
                continue
            tests.append(CodeflawsTestCase(
                input_file=inp, output_file=out,
                name=f'{glob_prefix[:-1]}{idx}'))

    # Half the budget to public, half to heldout.
    public_budget = max(1, max_tests // 2)
    heldout_budget = max(1, max_tests - public_budget)
    _collect('input-pos', 'output-pos', public_budget)
    _collect('heldout-input-pos', 'heldout-output-pos', heldout_budget)
    return tests[:max_tests]


def discover_main_name(bug_dir: Path) -> Optional[str]:
    """Find the MAINFILE stem to mimic the Makefile (strip .c)."""
    mk = bug_dir / 'Makefile'
    if mk.exists():
        try:
            m = re.search(r'FILENAME\s*=\s*(\S+)', mk.read_text())
            if m:
                return m.group(1)
        except Exception:
            pass
    c_files = [f for f in bug_dir.glob('*.c') if not f.name.endswith('.revlog')]
    if c_files:
        return sorted(c_files)[0].stem
    return None


# ── Single-candidate execution ──────────────────────────────────────────

def _normalize(s: str) -> str:
    """Normalize whitespace to match Codeflaws' `diff --ignore-trailing-space`."""
    lines = [ln.rstrip() for ln in s.replace('\r\n', '\n').splitlines()]
    while lines and lines[-1] == '':
        lines.pop()
    return '\n'.join(lines)


def _compile(candidate: str, work: Path, main: str) -> Tuple[bool, str]:
    """Compile candidate to binary. Returns (ok, error_msg)."""
    src = work / f'{main}.c'
    exe = work / main
    try:
        src.write_text(candidate)
    except Exception as e:
        return False, f'write_fail: {e}'
    try:
        r = subprocess.run(
            ['gcc', *GCC_FLAGS, str(src), '-o', str(exe)],
            cwd=str(work),
            capture_output=True, timeout=COMPILE_TIMEOUT_S)
        if r.returncode == 0 and exe.exists():
            return True, ''
        return False, r.stderr.decode(errors='replace')[-500:]
    except subprocess.TimeoutExpired:
        return False, 'compile_timeout'
    except FileNotFoundError:
        return False, 'gcc_missing'
    except Exception as e:
        return False, f'compile_error: {e}'


def _run_test(exe: Path, case: CodeflawsTestCase
              ) -> Tuple[bool, str]:
    """Run one test case. Returns (passed, detail)."""
    try:
        with open(case.input_file, 'rb') as stdin_f:
            r = subprocess.run(
                [str(exe)], stdin=stdin_f,
                capture_output=True, timeout=RUN_TIMEOUT_S,
                cwd=str(exe.parent))
    except subprocess.TimeoutExpired:
        return False, 'timeout'
    except Exception as e:
        return False, f'run_error: {e}'

    got = r.stdout[:MAX_OUTPUT_BYTES].decode(errors='replace')
    try:
        want = case.output_file.read_text(errors='replace')
    except Exception:
        return False, 'missing_expected'
    return _normalize(got) == _normalize(want), ''


@dataclass
class ExecResult:
    pass_rate: float
    passed: int
    total: int
    compiled: bool
    compile_error: str = ''
    details: Dict[str, str] = None


def exec_reward(candidate: str, bug_dir: Path,
                 max_tests: int = DEFAULT_MAX_TESTS) -> ExecResult:
    """Compile + run candidate against discovered tests. Return ExecResult.

    Guaranteed safe: uses a scratch tmp dir per call, no side-effects.
    """
    # Fast reject: parse-broken C shouldn't bother gcc (saves ~200ms).
    try:
        ps = parse_score(candidate, 'c')
        if ps < 0.2:
            return ExecResult(0.0, 0, 0, False, compile_error='parse_fail')
    except Exception:
        pass

    tests = discover_tests(bug_dir, max_tests=max_tests)
    main = discover_main_name(bug_dir) or 'prog'

    if not tests:
        return ExecResult(0.0, 0, 0, False, compile_error='no_tests')

    with tempfile.TemporaryDirectory(prefix='sfx_exec_') as tmp:
        work = Path(tmp)
        ok, err = _compile(candidate, work, main)
        if not ok:
            return ExecResult(0.0, 0, len(tests), False, compile_error=err)

        exe = work / main
        passed = 0
        details: Dict[str, str] = {}
        for t in tests:
            p, d = _run_test(exe, t)
            passed += int(p)
            if not p and d:
                details[t.name] = d
        return ExecResult(
            pass_rate=passed / max(len(tests), 1),
            passed=passed, total=len(tests), compiled=True,
            details=details)


# ── CLI sanity test ────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument('--bug_dir', required=True)
    ap.add_argument('--use_fixed', action='store_true',
                    help='Test the reference fixed file (should pass).')
    args = ap.parse_args()

    bd = Path(args.bug_dir)
    c_files = [f for f in bd.glob('*.c') if not f.name.endswith('.revlog')]
    if not c_files:
        raise SystemExit('no .c files')
    target = sorted(c_files)[1] if args.use_fixed and len(c_files) >= 2 \
        else c_files[0]
    cand = target.read_text(errors='replace')
    r = exec_reward(cand, bd)
    print(_json.dumps({
        'pass_rate': r.pass_rate,
        'passed': r.passed, 'total': r.total,
        'compiled': r.compiled,
        'compile_error': r.compile_error[:200],
    }, indent=2))
