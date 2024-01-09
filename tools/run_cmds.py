#!/usr/bin/env python

import subprocess
import argparse
from multiprocessing.pool import ThreadPool as Pool
import sys
import os
from signal import SIGINT
import traceback

def run_command(*command):
    subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        for line in sys.stdin:
            p.apply_async(run_command, (line,), error_callback=exit_with_error)

if __name__ == '__main__':
    main()
