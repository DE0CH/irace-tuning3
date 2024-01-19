#!/usr/bin/env python
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    for file in os.listdir(args.run_dir):
        if file.endswith('.log'):
            old_file = file
            new_file = file.split('.toml-')[1]
            print(f'rename {old_file} to {new_file}')
            if not args.dry_run:
                os.rename(old_file, new_file)

if __name__ == '__main__':
    main()
