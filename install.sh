set -euo pipefail

uv venv --seed --clear
uv sync
uv run bash -lc 'cd AReaL && bash examples/env/setup-pip-deps.sh'
uv pip uninstall pynvml