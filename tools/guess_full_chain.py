#!/usr/bin/env python

import subprocess
import argparse
import toml
import os
import json
import sys
from copy import deepcopy
from pathlib import Path

def set_deps(deps, key, value):
    if not key in deps:
        deps[key] = value

def run_cmd(cmd):
    return lambda: subprocess.run(cmd, check=True)

def find_cutoff_and_wallclock_limit(file, deps):
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('cutoff_time'):
                cutoff = line.split('=')[1].strip()
            elif line.startswith('wallclock-limit'):
                wallclock = line.split('=')[1].strip()
    set_deps(deps, 'cutoff', cutoff)
    set_deps(deps, 'wallclock', wallclock)

def split_instances(deps):
    with open(f"target_algorithms/surrogate/{deps['name']}/inst_feat_dict.{deps['name']}.par10.random.json", encoding='utf-8') as f:
        instances = json.load(f)
    po = list(instances.keys())
    with open(f"target_instances/surrogate/{deps['name']}/train-1.txt", 'w', encoding='utf-8') as f:
        for i in po[:len(po)//3]:
            f.write(f"{i}\n")
    with open(f"target_instances/surrogate/{deps['name']}/test-1.txt", 'w', encoding='utf-8') as f:
        for i in po[len(po)//3:len(po)//3*2]:
            f.write(f"{i}\n")
    with open(f"target_instances/surrogate/{deps['name']}/validation-1.txt", 'w', encoding='utf-8') as f:
        for i in po[len(po)//3*2:]:
            f.write(f"{i}\n")

def write_tomls(deps):
    name = deps['name']
    wallclock = deps['wallclock']
    cutoff = deps['cutoff']
    args = {
        "surrogate_args": {
            "pyrfr_wrapper": f"target_algorithms/surrogate/{name}/pyrfr_wrapper.{name}.par10.random.pkl",
            "pyrfr_model": f"target_algorithms/surrogate/{name}/pyrfr_model.{name}.par10.random.bin",
            "config_space": f"target_algorithms/surrogate/{name}/config_space.{name}.par10.random.pcs",
            "inst_feat_dict": f"target_algorithms/surrogate/{name}/inst_feat_dict.{name}.par10.random.json"
        },
        "irace_args": {
            "parameter_file": f"target_instances/surrogate/{name}/config_space.{name}.par10.random.irace.txt",
            "train_instances_file": f"target_instances/surrogate/{name}/train-1.txt",
            "test_instances_file": f"target_instances/surrogate/{name}/test-1.txt",
            "validation_instances_file": f"target_instances/surrogate/{name}/validation-1.txt",
            "digits": 10,
            "max_time": wallclock,
            "bound_max": cutoff,
        }
    }
    Path(args['surrogate_args']['pyrfr_wrapper']).resolve(strict=True)
    Path(args['surrogate_args']['pyrfr_model']).resolve(strict=True)
    Path(args['surrogate_args']['config_space']).resolve(strict=True)
    Path(args['surrogate_args']['inst_feat_dict']).resolve(strict=True)
    Path(args['irace_args']['parameter_file']).resolve(strict=True)
    Path(args['irace_args']['train_instances_file']).resolve(strict=True)
    Path(args['irace_args']['test_instances_file']).resolve(strict=True)
    Path(args['irace_args']['validation_instances_file']).resolve(strict=True)

    train_args = deepcopy(args)
    del train_args['irace_args']['validation_instances_file']

    with open(f'Instances/{deps["name"]}-1.toml', 'w', encoding='utf-8') as f:
        toml.dump(train_args, f)

    test_args = deepcopy(args)
    test_args['irace_args']['test_instances_file'] = test_args['irace_args']['validation_instances_file']
    del test_args['irace_args']['validation_instances_file']
    with open(f'Instances-test/{deps["name"]}-1.toml', 'w', encoding='utf-8') as f:
        toml.dump(test_args, f)

    with open(f'jobs/{deps["name"]}/instances.txt', 'w', encoding='utf-8') as f:
        f.write(f"{deps['name']}-1.toml\n")
    with open(f'jobs/{deps["name"]}/instances-test.txt', 'w', encoding='utf-8') as f:
        f.write(f"{deps['name']}-1.toml\n")


def write_jobs_scripts(deps):
    with open(f'jobs/{deps["name"]}/gen.sh', 'w', encoding='utf-8') as f:
        f.write(f"""#!/bin/sh -e

export IRACE_TUNING_PATH="${{IRACE_TUNING_PATH:-../irace-tuning3}}"

"$IRACE_TUNING_PATH/tools/make_job_sh.py" \\
    --irace-tuning-run-dir '${{IRACE_TUNING_RUN_DIR:-runs/{deps["name"]}}}' \\
    --train-instances-file jobs/{deps["name"]}/instances.txt \\
    --test-instances-dir Instances-test \\
    --test-instances-file jobs/{deps["name"]}/instances-test.txt \\
    --target-runner '${{IRACE_TUNING_PATH}}/target-runner.py' \\
    --log-file '${{IRACE_TUNING_RUN_DIR}}/train/irace.Rdata' \\
    --parameter-file jobs/parameters.txt \\
    --max-experiments 3000 --seed 123 \\
    --parallel '${{IRACE_TUNING_NCPU:-1}}' \\
""")
    os.chmod(f'jobs/{deps["name"]}/gen.sh', os.stat(f'jobs/{deps["name"]}/gen.sh').st_mode | 0o111)
    with open(f'jobs/{deps["name"]}/run.sh', 'w', encoding='utf-8') as f:
        subprocess.run([f'./jobs/{deps["name"]}/gen.sh'], stdout=f, check=True)
    os.chmod(f'jobs/{deps["name"]}/run.sh', os.stat(f'jobs/{deps["name"]}/run.sh').st_mode | 0o111)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("epm_train_url")
    parser.add_argument("surrogate_url")
    parser.add_argument("deps_file")
    parser.add_argument("--step", "-s", default=0, type=int)
    args = parser.parse_args()
    try:
        with open(args.deps_file, encoding='utf-8') as f:
            deps = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        deps = {}
    set_deps(deps, 'name', args.epm_train_url.split('/')[-1].split('.')[0])

    steps = [
        run_cmd(['curl', '-OJL', args.epm_train_url]),
        run_cmd(['tar', 'xvf', args.epm_train_url.split('/')[-1]]),
        run_cmd(['curl', '-OJL', args.surrogate_url]),
        run_cmd(['unzip', args.surrogate_url.split('/')[-1]]),
        lambda: find_cutoff_and_wallclock_limit(deps['name']+'/scenario.txt', deps),
        run_cmd(['mkdir', '-p', f"target_instances/surrogate/{deps['name']}"]),
        run_cmd([sys.executable, '../irace-tuning3/tools/convert_pcs_to_irace.py', f"target_algorithms/surrogate/{deps['name']}/config_space.{deps['name']}.par10.random.pcs", f"target_instances/surrogate/{deps['name']}/config_space.{deps['name']}.par10.random.irace.txt"]),
        lambda: split_instances(deps),
        run_cmd(['mkdir', '-p', f"jobs/{deps['name']}"]),
        lambda: write_tomls(deps),
        lambda: write_jobs_scripts(deps),
    ]

    for i in range(args.step, len(steps)):
        try:
            steps[i]()
        except Exception as exc:
            with open(args.deps_file, 'w', encoding='utf-8') as f:
                json.dump(deps, f, indent=4)
            raise RuntimeError(f"Failed to run step {i}") from exc


if __name__ == '__main__':
    main()
