#!/usr/bin/env bash
# SynthFix pipeline entry point.
#
# Runs the final SynthFix pipeline end-to-end:
#   1. Orchestrate all (model, dataset) pairs via orchestrate_final.py
#   2. Aggregate per-run JSONs into results/final/final_report.{json,md}
#
# Optional first argument: PID of a gate job to wait on before starting
# (e.g. when staged after a long SFT warmup on a shared GPU).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

mkdir -p results/final
ORCH_LOG="results/final/orchestrator.log"
AGG_LOG="results/final/aggregate.log"

GATE_PID="${1:-}"
if [ -n "$GATE_PID" ] && kill -0 "$GATE_PID" 2>/dev/null; then
  echo "[$(date +%T)] Waiting for gate PID $GATE_PID..." | tee -a "$ORCH_LOG"
  while kill -0 "$GATE_PID" 2>/dev/null; do sleep 60; done
fi

echo "[$(date +%T)] Running orchestrator..." | tee -a "$ORCH_LOG"
python3 -u orchestrate_final.py >> "$ORCH_LOG" 2>&1
RC=$?
echo "[$(date +%T)] Orchestrator exit=$RC" | tee -a "$ORCH_LOG"

echo "[$(date +%T)] Aggregating..." | tee -a "$ORCH_LOG"
python3 -u aggregate_final.py >> "$AGG_LOG" 2>&1

echo "[$(date +%T)] Pipeline done (orch rc=$RC)." | tee -a "$ORCH_LOG"
exit $RC
