#!/usr/bin/env python
import toml
import os
import sys
import subprocess
import argparse
import subprocess
import signal
import sys
import shlex
import os
import time
from watchdog.events import FileSystemEventHandler
from watchdog.events import FileCreatedEvent
from threading import Event
import pickle
import requests

def extract_from_logfile(logfile):
    quoted_logfile = repr(logfile)
    command = ['Rscript', '-e', f"load({quoted_logfile}); cat(mean(iraceResults$testing$experiment))"]
    res = subprocess.check_output(command, stderr=subprocess.DEVNULL).decode('utf-8').strip()
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
    with open(instance, 'r', encoding='utf-8') as f:
        instance = toml.load(f)

    server_args = [
        sys.executable,
        '-u',
        '-m', 'epm.webserver.flask_server',
        '--pyrfr_wrapper', os.path.abspath(instance['surrogate_args']['pyrfr_wrapper']),
        '--pyrfr_model', os.path.abspath(instance['surrogate_args']['pyrfr_model']),
        '--config_space', os.path.abspath(instance['surrogate_args']['config_space']),
        '--inst_feat_dict', os.path.abspath(instance['surrogate_args']['inst_feat_dict']),
        '--pid', '0', # We can also use pid to differentiate but it is not necessary as we use dir.
        '--dir', '.',
        '--nic_name', 'localhost',
    ]

    irace_args = [
        os.path.join(subprocess.check_output(['Rscript', '-e', "cat(system.file(package=\'irace\', \'bin\', mustWork=TRUE))"]).decode('utf-8'), 'irace'),
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
        irace_args.extend(['--max-experiments', str(instance['irace_args']['max_experiments'])])
    if 'max_time' in instance['irace_args']:
        irace_args.extend(['--max-time', str(instance['irace_args']['max_time'])])

    os.makedirs(os.path.join(IRACE_TUNING_RUN_DIR, run_name), exist_ok=True)
    os.chdir(os.path.join(IRACE_TUNING_RUN_DIR, run_name))
    if os.path.isfile(os.path.join(IRACE_TUNING_RUN_DIR, run_name, 'irace.Rdata')):
        print(extract_from_logfile(os.path.join(IRACE_TUNING_RUN_DIR, run_name, 'irace.Rdata')))
        return

    try:
        os.remove('./nameserver_creds.pkl')
    except OSError:
        pass
    with open('./server-log.log', 'w', encoding='utf-8', buffering=1) as f:
        server_p = subprocess.Popen(server_args + ['start'], stdout=f, stderr=f)
    with open('./server.pid', 'w', encoding='utf-8') as f:
        print(server_p.pid, file=f)

    while True:
        try:
            with open('nameserver_creds.pkl', 'rb') as f:
                ip, port, _ = pickle.load(f)
            response = requests.get(f'http://{ip}:{port}/status')
            if response.status_code == 200:
                break
        except (requests.exceptions.ConnectionError, FileNotFoundError):
            pass
        finally:
            time.sleep(0.1)
        if server_p.poll() is not None:
            raise RuntimeError('Server stopped unexpectedly.')

    irace_args.extend(algs_options)
    with open('irace-log.log', 'w', encoding='utf-8', buffering=1) as f:
        irace_p = subprocess.run(irace_args, stdout=f, stderr=f, check=False)
    server_p.send_signal(signal.SIGINT)

    if irace_p.returncode != 0:
        print('inf')
    else:
        print(extract_from_logfile('irace.Rdata'))

if __name__ == '__main__':
    main()
