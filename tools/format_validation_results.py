#!/usr/bin/env python

import argparse
import json
import random
import itertools
import sys
from generate_validation_commands import get_best_configurations

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
    seeds = [str(rng.randrange(10000000, 1000000000)) for _ in range(int(args.n_seed))]
    results = map(lambda x:
        {
            'configuration_id': x[0],
            'instance_id': x[1],
            'configuration': configurations[x[0]],
            'instance': instances[x[1]],
            'seed': x[2],
            'test_instance': instances[x[1]],
            'test_instance_dir': args.test_instances_dir,
        },
        itertools.product(
        range(len(configurations)),
        range(len(instances)),
        seeds,
    ))
    json.dump(list(results), indent=4, fp=sys.stdout)

if __name__ == '__main__':
    main()
