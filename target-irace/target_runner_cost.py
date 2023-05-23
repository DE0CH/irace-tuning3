#!/usr/bin/env python
import epm.webserver.flask_worker
import sys
import argparse

def main():
    configuration_id = int(sys.argv[1])
    instance_id = sys.argv[2]
    seed = int(sys.argv[3])
    instance = sys.argv[4]
    bound_max = int(sys.argv[5])
    algs_options = sys.argv[6:]
    
    new_args = [
        '--dir', '.',
        '--instance_name', instance[2:], # We ignore the first two slashes as a workaround.
        '--cutoff', '1000000000000', #TODO: Figure out how to set this to inf
        '--seed', str(seed),
        *algs_options
    ]
    cost, _ = epm.webserver.flask_worker.send_procedure(new_args) 
    print(cost[0])

    


if __name__ == '__main__':
    main()
