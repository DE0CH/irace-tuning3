import argparse

parser = argparse.ArgumentParser()
parser.add_argument('irace_log', type=str, help='irace log file')
args = parser.parse_args()

cmds = []

with open(args.irace_log, 'r') as f:
    for line in f:
        if '/workspaces' in line:
            cmds.append(line[line.index('/workspaces'):].strip())

print('\n'.join(cmds))