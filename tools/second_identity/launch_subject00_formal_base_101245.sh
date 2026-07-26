#!/usr/bin/env bash
set -euo pipefail

TASK_ID="AAAI27-SUBJECT00-FORMAL-BASE-101245-001"
SESSION="subject00_formal_base_101245_001"
WT="/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_subject00_formal_base_101245_execution"
PY="/root/autodl-tmp/conda_envs/mmlphuman/bin/python"
OUTPUT_ROOT="/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-BASE-101245-001"
RUN_ROOT="${OUTPUT_ROOT}/attempt_001"
SPOOL="/root/autodl-tmp/canondressgs_work/runtime_logs/${TASK_ID}/attempt_001"
COMMAND="${PY} tools/second_identity/run_subject00_formal_base_101245.py --phase train --attempt-root ${RUN_ROOT}"

cd "${WT}"

if [ -e "${OUTPUT_ROOT}" ]; then
  echo "output root already exists: ${OUTPUT_ROOT}" >&2
  exit 10
fi

if nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | grep -q .; then
  echo "GPU compute process already active" >&2
  exit 11
fi

"${PY}" tools/second_identity/validate_subject00_formal_base_launch_gate.py
"${PY}" -m pytest tests/test_subject00_formal_base_launch_gate.py
"${PY}" tools/second_identity/run_subject00_formal_base_101245.py --phase preflight --attempt-root "${RUN_ROOT}" > /tmp/subject00_formal_base_101245_preflight.json

mkdir -p "${SPOOL}"
printf '%s\n' "${COMMAND}" > "${SPOOL}/exact_command.txt"
env | sort > "${SPOOL}/env_snapshot.txt"
cat > "${SPOOL}/run_train.sh" <<RUN
#!/usr/bin/env bash
set -euo pipefail
cd "${WT}"
exec ${COMMAND}
RUN
chmod +x "${SPOOL}/run_train.sh"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 12
  fi
  tmux new-session -d -s "${SESSION}" -c "${WT}" "bash '${SPOOL}/run_train.sh' >'${SPOOL}/stdout.log' 2>'${SPOOL}/stderr.log'"
  LAUNCHER="tmux"
  PID="$(tmux list-panes -t "${SESSION}" -F "#{pane_pid}" | head -n 1)"
else
  nohup bash "${SPOOL}/run_train.sh" > "${SPOOL}/stdout.log" 2> "${SPOOL}/stderr.log" &
  PID="$!"
  LAUNCHER="nohup"
fi

for _ in $(seq 1 120); do
  if [ -d "${RUN_ROOT}" ]; then
    break
  fi
  sleep 1
done

if [ ! -d "${RUN_ROOT}" ]; then
  echo "run root was not created within 120 seconds: ${RUN_ROOT}" >&2
  exit 13
fi

mkdir -p "${RUN_ROOT}/logs"
ln -sfn "${SPOOL}/stdout.log" "${RUN_ROOT}/logs/stdout.log"
ln -sfn "${SPOOL}/stderr.log" "${RUN_ROOT}/logs/stderr.log"
ln -sfn "${SPOOL}/exact_command.txt" "${RUN_ROOT}/logs/exact_command.txt"
ln -sfn "${SPOOL}/env_snapshot.txt" "${RUN_ROOT}/logs/env_snapshot.txt"
ln -sfn "../training_logs/step_records.jsonl" "${RUN_ROOT}/logs/progress_step_records.jsonl"

cat > "${SPOOL}/session_metadata.json" <<JSON
{
  "task_id": "${TASK_ID}",
  "session_name": "${SESSION}",
  "launcher": "${LAUNCHER}",
  "pid": "${PID}",
  "working_directory": "${WT}",
  "run_root": "${RUN_ROOT}",
  "command": "${COMMAND}",
  "launched_at_unix": $(date +%s)
}
JSON
ln -sfn "${SPOOL}/session_metadata.json" "${RUN_ROOT}/logs/session_metadata.json"

cat "${RUN_ROOT}/logs/session_metadata.json"
