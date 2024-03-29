#!/usr/bin/env python

import argparse
import json
import random
import itertools
import sys
from generate_validation_commands import get_best_configurations
from os import path

def main():
    rng = random.Random(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-instances-file', required=True)
    parser.add_argument('--test-instances-dir', required=True)
    parser.add_argument('--log-file', required=True)
    parser.add_argument('--n-seed', required=True)
    parser.add_argument('--test-log-file', required=True)
    args = parser.parse_args()
    with open(args.test_instances_file, 'r', encoding='utf-8') as f:
        instances = f.read().splitlines()
    configurations = [''] + list(get_best_configurations(args.log_file))
    seeds = [str(rng.randrange(10000000, 1000000000)) for _ in range(int(args.n_seed) * len(configurations) * len(instances))]
    with open(args.test_log_file, encoding='utf-8') as f:
        results_numbers = f.read().splitlines()
    results = map(lambda y:
        {
            'configuration_id': y[1][0] + 1,
            'instance_id': y[1][1] + 1,
            'configuration': configurations[y[1][0]],
            'instance': path.join(args.test_instances_dir, instances[y[1][1]]),
            'seed': seeds[y[0]],
            'cost': results_numbers[y[0]],
        },
        enumerate(itertools.product(
        range(len(configurations)),
        range(len(instances)),
        range(int(args.n_seed)),
    )))
    json.dump(list(results), indent=4, fp=sys.stdout)

if __name__ == '__main__':
    main()
