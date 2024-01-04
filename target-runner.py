#!/usr/bin/env python
import argparse
import toml
import os
import sys
import subprocess
import shlex

def extract_from_logfile(logfile):
    quoted_logfile = repr(logfile)
    command = ['Rscript', '-e', f"load({quoted_logfile}); cat(mean(iraceResults$testing$experiment))"]
    res = subprocess.check_output(command).decode('utf-8').strip()
    try:
        float(res)
        return res
    except ValueError:
        return 'inf'

def get_abs_path(path):
    return os.path.abspath(path)

def main():
    IRACE_TUNING_PATH = os.environ.get('IRACE_TUNING_PATH', os.path.dirname(os.path.realpath(__file__)))
    IRACE_TUNING_RUN_DIR = os.environ.get('IRACE_TUNING_RUN_DIR', 'runs')
    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    instance = sys.argv[4]
    algs_options = sys.argv[5:]

    run_name = f"{os.path.basename(os.path.normpath(instance))}-{configuration_id}-{instance_id}-{seed}"
    with open(instance, 'r') as f:
        instance = toml.load(f)

    target_args = [
        '--pyrfr_wrapper', get_abs_path(instance['surrogate_args']['pyrfr_wrapper']),
        '--pyrfr_model', get_abs_path(instance['surrogate_args']['pyrfr_model']),
        '--config_space', get_abs_path(instance['surrogate_args']['config_space']),
        '--inst_feat_dict', get_abs_path(instance['surrogate_args']['inst_feat_dict']),
        '--',
        '--target-runner', get_abs_path(os.path.join(IRACE_TUNING_PATH, 'target-irace/target_runner/target/release/target_runner')),
        '--parameter-file', get_abs_path(instance['irace_args']['parameter_file']),
        '--train-instances-file', get_abs_path(instance['irace_args']['train_instances_file']),
        '--train-instances-dir', '/', #FIXME: this is because otherwise it will be relative to the cwd of irace.
        '--test-instances-file', get_abs_path(instance['irace_args']['test_instances_file']),
        '--test-instances-dir', '/', #FIXME: this is because otherwise it will be relative to the cwd of irace.
        '--digits', str(instance['irace_args']['digits']),
        '--seed', str(seed),
        '--bound-max', str(instance['irace_args']['bound_max']),
    ]

    if 'max_experiments' in instance['irace_args']:
        target_args.extend(['--max-experiments', str(instance['irace_args']['max_experiments'])])
    if 'max_time' in instance['irace_args']:
        target_args.extend(['--max-time', str(instance['irace_args']['max_time'])])

    target_args.extend(algs_options)
    start_py_path = get_abs_path(os.path.join(IRACE_TUNING_PATH, 'target-irace/start.py'))
    os.makedirs(os.path.join(IRACE_TUNING_RUN_DIR, run_name), exist_ok=True)
    if not os.path.isfile(os.path.join(IRACE_TUNING_RUN_DIR, run_name, 'irace.Rdata')):
        target_irace = subprocess.Popen([sys.executable, "-u", start_py_path, *target_args], cwd=os.path.join(IRACE_TUNING_RUN_DIR, run_name), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        subprocess.Popen(['tee', os.path.join(IRACE_TUNING_RUN_DIR, run_name, 'irace-log.txt')], stdin=target_irace.stdout, stdout=subprocess.DEVNULL)
        target_irace.wait()
    if target_irace.returncode != 0:
        print('inf')
    else:
        print(extract_from_logfile(os.path.join(IRACE_TUNING_RUN_DIR, run_name, 'irace.Rdata')))

if __name__ == '__main__':
    main()
