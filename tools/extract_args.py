import argparse
import shlex

parser = argparse.ArgumentParser()
parser.add_argument('shell_file', type=str, help='Path to the shell file.')
args = parser.parse_args()
with open(args.shell_file, 'r') as f:
    shell_file = f.read()
print(' '.join(shlex.split(shell_file)[1:]))