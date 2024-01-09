#!/usr/bin/env python

import subprocess
import argparse
from multiprocessing.pool import ThreadPool as Pool
import sys
import os
from signal import SIGINT
import traceback

def run_command(*command):
    p = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE)
    return p.stdout.decode('utf-8').strip()

def exit_with_error(e):
    try:
        raise e
    #pylint: disable=bare-except
    except:
        print(traceback.format_exc())
    os.kill(os.getpid(), SIGINT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()
    with Pool(args.parallel) as p:
        cmds = [line for line in sys.stdin]
        res = p.map_async(run_command, cmds, error_callback=exit_with_error)
        res = res.get()
        p.close()
        p.join()
    print(*res, sep='\n')
if __name__ == '__main__':
    main()
