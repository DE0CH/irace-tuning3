#!/usr/bin/env python
import epm.webserver.flask_worker
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration_id', type=int, help='The configuration id.')
    parser.add_argument('instance_id', type=int, help='The instance id.')
    parser.add_argument('seed', type=int, help='The seed.')
    parser.add_argument('instance', type=str, help='The instance.')
    parser.add_argument('bound_max', type=int, help='The maximum bound.')
    parser.add_argument('algs_options', nargs=argparse.REMAINDER, help='The options for the algorithm.')
    args = parser.parse_args()
    
    new_args = [
        '--dir', '.',
        '--instance_name', args.instance[2:], # We ignore the first two slashes as a workaround.
        '--cutoff', '1000000000000', #TODO: Figure out how to set this to inf
        '--seed', str(args.seed),
        *args.algs_options
    ]
    cost, _ = epm.webserver.flask_worker.send_procedure(new_args) 
    print(cost[0])

    


if __name__ == '__main__':
    main()
