#!/usr/bin/env python

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--irace-tuning-run-dir", required=True)
    parser.add_argument("--train-instances-dir", default="Instances")
    parser.add_argument("--train-instances-file", required=True)
    parser.add_argument("--test-instances-dir", default="Instances-test")
    parser.add_argument("--test-instances-file", required=True)
    parser.add_argument("--target-runner", default="${IRACE_TUNING_PATH}/target-runner.py")
    parser.add_argument("--log-file", default="${IRACE_TUNING_RUN_DIR}/train/irace.Rdata")
    parser.add_argument("--parameter-file", default="jobs/parameters.txt")
    parser.add_argument("--max-experiments", required=True)
    parser.add_argument("--seed", default="123")
    parser.add_argument("--parallel", default="${IRACE_TUNING_NCPU:-1}")
    parser.add_argument("--irace-log", default="${IRACE_TUNING_RUN_DIR}/train/irace-log.log")
    args = parser.parse_args()
    print(f"""#!/bin/sh -e

# Run this script with cwd in the project root folder
# generated by make-job-sh.py

export IRACE_TUNING_PATH="${{IRACE_TUNING_PATH:-../irace-tuning3}}"
export IRACE_TUNING_RUN_DIR="{args.irace_tuning_run_dir}"

mkdir -p "${{IRACE_TUNING_RUN_DIR}}/train"
IRACE_TUNING_RUN_DIR="${{IRACE_TUNING_RUN_DIR}}/train" irace \\
    --train-instances-dir "{args.train_instances_dir}" \\
    --train-instances-file "{args.train_instances_file}" \\
    --test-instances-dir "{args.test_instances_dir}" \\
    --test-instances-file "{args.test_instances_file}" \\
    --target-runner "{args.target_runner}" \\
    --log-file "{args.log_file}" \\
    --parameter-file "{args.parameter_file}" \\
    --max-experiment {args.max_experiments} --seed 123 \\
    --parallel {args.parallel} \\
    | tee "{args.irace_log}"
""")

if __name__ == "__main__":
    main()
