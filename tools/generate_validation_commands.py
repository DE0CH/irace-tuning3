#!/usr/bin/env python

import argparse
import itertools
import random
from os.path import abspath
from os.path import join as path_join
import shlex


def get_best_configurations(logfile):
    with open(logfile, 'r', encoding='utf-8') as f:
        active = False
        for line in f:
            if line.startswith('# Best configurations as commandlines'):
                active = True
            elif active and line.startswith('#'):
                active = False
            elif active and line.strip():
                yield line.split(' ', 1)[1].strip()

def main():
    rng = random.Random(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-instances-file', required=True)
    parser.add_argument('--test-instances-dir', required=True)
    parser.add_argument('--target-runner', required=True)
    parser.add_argument('--n-seed', required=True)
    parser.add_argument('--log-file', required=True)
    parser.add_argument('--parallel', default='1')

    args = parser.parse_args()

    with open(args.test_instances_file, 'r', encoding='utf-8') as f:
        instances = f.read().splitlines()
    configurations = [''] + list(get_best_configurations(args.log_file))
    seeds = [str(rng.randrange(10000000, 1000000000)) for _ in range(int(args.n_seed) * len(configurations) * len(instances))]
    cmds = itertools.product(
        range(len(configurations)),
        range(len(instances)),
        range(int(args.n_seed)),
    )

    cmds = map(lambda y: f'{abspath(args.target_runner)} {y[1][0]+1} {y[1][1]+1} {seeds[y[0]]} {shlex.quote(abspath(path_join(args.test_instances_dir,instances[y[1][1]])))} {configurations[y[1][0]]}', enumerate(cmds))
    for cmd in cmds:
        print(cmd)

if __name__ == '__main__':
    main()
