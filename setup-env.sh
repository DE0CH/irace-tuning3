#!/usr/bin/env bash
# activate venv and sets up environment variables
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) &&
source "$SCRIPT_DIR/venv/bin/activate" &&
export PATH="$SCRIPT_DIR/target-irace/target_runner/target/release:$PATH" &&
export PYTHONPATH="$SCRIPT_DIR/epm" &&
export workspaceFolder=$SCRIPT_DIR
