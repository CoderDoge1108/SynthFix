# Data Directory

This directory is intentionally a placeholder in the public artifact.

The scripts expect benchmark data in the following locations when reproducing experiments:

```text
data/benchmarks_processed/pyrepair
data/benchmarks_processed/sven
data/raw_benchmarks/
artifact/work/codeflaws_exec
```

Use `python artifact/build_pyexec_benchmark.py` to build pyrepair from the MBPP/QuixBugs inputs. CodeFlaws and SVEN should be prepared following their original licenses and placed at the paths above.
