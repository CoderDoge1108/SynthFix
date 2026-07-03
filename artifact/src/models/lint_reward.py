"""
SynthFix-Exec: language-specific static linter rewards.

Three language-specific signals that complement the tree-sitter parse
score with real *semantic* static-analysis errors (undefined refs,
missing imports, type confusion, etc.). These are orthogonal to
surface-similarity (chrF / CodeBLEU) so they give the RL agent and the
reranker a genuinely new signal to chase.

All checks run fast (~1-50 ms) in subprocesses and are reference-free.
They return a [0, 1] *quality* score where 1.0 = no issues found.

  * lint_python(code):  pyflakes (undefined refs, unused imports,
                        redefined names). Converts error count to
                        quality via exp-decay.
  * lint_c(code):       gcc -Wall -Wextra -fsyntax-only. Compile-only
                        check; counts diagnostic lines.
  * lint_javascript(code): tree-sitter undefined-identifier heuristic —
                        identifiers referenced but never declared in
                        params / var / const / let / function / import.

For each: empty or whitespace-only code returns 0.0.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# ── Python: pyflakes ────────────────────────────────────────────────────

try:
    from pyflakes.api import check as _pyflakes_check
    import io
    _HAS_PYFLAKES = True
except Exception:
    _HAS_PYFLAKES = False


def _lint_python_api(code: str) -> int:
    """Return number of pyflakes issues using the in-process API. -1 on error."""
    if not code.strip():
        return -1
    if not _HAS_PYFLAKES:
        return -1
    try:
        sio = io.StringIO()
        n = _pyflakes_check(code, '<mem>', reporter=_PyflakesReporter(sio))
        return int(n)
    except Exception:
        return -1


class _PyflakesReporter:
    """Silent reporter that swallows output and just counts issues."""

    def __init__(self, _stream):
        self._stream = _stream

    def unexpectedError(self, filename, msg):
        pass

    def syntaxError(self, filename, msg, lineno, offset, text):
        pass

    def flake(self, message):
        pass


def lint_python(code: str) -> float:
    """Python lint reward in [0, 1]. 1.0 = no pyflakes issues."""
    n = _lint_python_api(code)
    if n < 0:
        return 0.5  # unknown: neutral
    # Exponential decay: 0 errors → 1.0, 5 → 0.37, 10 → 0.14
    import math
    return math.exp(-n / 5.0)


# ── C: gcc -Wall -fsyntax-only ──────────────────────────────────────────

_GCC_NOTE_RE = re.compile(r':\s*(error|warning):', re.I)


def _lint_c_subproc(code: str, timeout: float = 5.0) -> int:
    """Return count of errors+warnings from gcc. -1 on failure."""
    if not code.strip():
        return -1
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c',
                                          delete=False) as f:
            f.write(code)
            fpath = f.name
        try:
            r = subprocess.run(
                ['gcc', '-Wall', '-Wextra', '-fsyntax-only',
                 '-std=c99', fpath],
                capture_output=True, timeout=timeout)
        finally:
            try: Path(fpath).unlink()
            except Exception: pass
        stderr = r.stderr.decode(errors='replace')
        return len(_GCC_NOTE_RE.findall(stderr))
    except Exception:
        return -1


def lint_c(code: str) -> float:
    """C lint reward in [0, 1]. 1.0 = no gcc warnings/errors."""
    n = _lint_c_subproc(code)
    if n < 0:
        return 0.5
    import math
    return math.exp(-n / 8.0)


# ── JavaScript: tree-sitter undefined-identifier heuristic ──────────────

try:
    from tree_sitter_languages import get_parser as _get_parser
    _JS_PARSER = _get_parser('javascript')
    _HAS_JS = True
except Exception:
    _JS_PARSER = None
    _HAS_JS = False


_JS_GLOBALS = {
    # Browser / node builtins that may appear "undefined" to our parser.
    'console', 'window', 'document', 'globalThis', 'self', 'process',
    'module', 'exports', 'require', 'Buffer', '__dirname', '__filename',
    'Math', 'Date', 'Array', 'Object', 'String', 'Number', 'Boolean',
    'JSON', 'RegExp', 'Error', 'TypeError', 'SyntaxError', 'Promise',
    'Map', 'Set', 'WeakMap', 'WeakSet', 'Symbol', 'Proxy', 'Reflect',
    'undefined', 'null', 'NaN', 'Infinity', 'isNaN', 'isFinite',
    'parseInt', 'parseFloat', 'encodeURI', 'encodeURIComponent',
    'decodeURI', 'decodeURIComponent', 'setTimeout', 'setInterval',
    'clearTimeout', 'clearInterval', 'fetch', 'Promise', 'async',
    'await', 'true', 'false', 'this', 'super', 'arguments', 'new',
    'typeof', 'instanceof', 'delete', 'void', 'in', 'of', 'yield',
    'Function',
}


def _collect_js_ident_and_decls(node, text: bytes,
                                 used: set, declared: set):
    """Walk JS AST, collecting declared names and identifier references.

    Decl binders (surface scan only — we don't track block scope):
      * function_declaration / class_declaration / method_definition:
            the *name* child (usually first ``identifier`` grandchild)
      * variable_declarator:  the LHS identifier
      * formal_parameters:    every identifier directly under it
      * import_specifier:     imported name
    Everything else that looks like a plain ``identifier`` in an
    expression context is a usage.
    """

    def _first_ident_child(n) -> Optional[str]:
        for ch in n.children:
            if ch.type == 'identifier' and ch.text:
                return ch.text.decode('utf8', errors='replace')
        return None

    stack = [node]
    while stack:
        n = stack.pop()
        t = n.type

        if t in ('function_declaration', 'class_declaration',
                 'method_definition', 'function'):
            name = _first_ident_child(n)
            if name:
                declared.add(name)
            # Recurse only into params + body; don't double-count name.
            for ch in n.children:
                if ch.type in ('formal_parameters', 'statement_block',
                                'class_body'):
                    stack.append(ch)
            continue

        if t == 'variable_declarator':
            name = _first_ident_child(n)
            if name:
                declared.add(name)
            # RHS is typically after '=' — recurse into everything that
            # isn't the initial identifier.
            seen_name = False
            for ch in n.children:
                if (not seen_name and ch.type == 'identifier'):
                    seen_name = True
                    continue
                stack.append(ch)
            continue

        if t == 'formal_parameters':
            # All top-level identifiers inside are declarations.
            for ch in n.children:
                if ch.type == 'identifier' and ch.text:
                    declared.add(ch.text.decode('utf8',
                                                 errors='replace'))
                else:
                    stack.append(ch)
            continue

        if t in ('import_specifier', 'import_clause'):
            for ch in n.children:
                if ch.type == 'identifier' and ch.text:
                    declared.add(ch.text.decode('utf8',
                                                 errors='replace'))
            continue

        # Property access — skip the property part (it's not an
        # identifier binding into our scope).
        if t == 'member_expression':
            for ch in n.children:
                if ch.type == 'property_identifier':
                    continue
                stack.append(ch)
            continue

        if t == 'identifier' and n.text:
            used.add(n.text.decode('utf8', errors='replace'))
            continue

        for ch in n.children:
            stack.append(ch)


def lint_javascript(code: str) -> float:
    """JS lint reward: fraction of used identifiers that are declared."""
    if not code.strip():
        return 0.0
    if not _HAS_JS:
        return 0.5
    try:
        tree = _JS_PARSER.parse(bytes(code, 'utf8', errors='replace'))
    except Exception:
        return 0.5
    used: set = set()
    declared: set = set()
    _collect_js_ident_and_decls(tree.root_node, bytes(code, 'utf8'),
                                 used, declared)
    if not used:
        return 1.0
    undeclared = used - declared - _JS_GLOBALS
    score = 1.0 - (len(undeclared) / max(len(used), 1))
    return max(0.0, min(1.0, score))


# ── Unified entry point ─────────────────────────────────────────────────

def lint_score(code: str, lang: str) -> float:
    """Language-aware lint reward in [0, 1]. 1.0 = clean."""
    if lang == 'python':
        return lint_python(code)
    if lang == 'c':
        return lint_c(code)
    if lang in ('javascript', 'js'):
        return lint_javascript(code)
    return 0.5


if __name__ == '__main__':
    good_py = 'def f(x):\n    return x + 1\n'
    bad_py = 'def f(x):\n    return y + 1\n'
    print('py good:', lint_python(good_py))
    print('py bad :', lint_python(bad_py))

    good_c = '#include <stdio.h>\nint main(){int x=0; printf("%d",x); return 0;}'
    bad_c = '#include <stdio.h>\nint main(){int x=0; printf(y); return 0;}'
    print('c  good:', lint_c(good_c))
    print('c  bad :', lint_c(bad_c))

    good_js = 'function f(x) { return x + 1; }'
    bad_js = 'function f(x) { return yy + 1; }'
    print('js good:', lint_javascript(good_js))
    print('js bad :', lint_javascript(bad_js))
