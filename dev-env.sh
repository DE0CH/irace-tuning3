#!/usr/bin/env bash
# env used for dev
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) &&
export PATH="$SCRIPT_DIR/target-irace/target_runner/target/release:$PATH" &&
export PYTHONPATH="$SCRIPT_DIR/epm" &&
export workspaceFolder=$SCRIPT_DIR
