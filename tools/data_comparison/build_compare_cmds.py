import argparse
import os
import shlex
import sys
import toml
parser = argparse.ArgumentParser()
parser.add_argument('input_dir')
parser.add_argument('output_dir')
parser.add_argument('data_comparison_dir')
parser.add_argument('target_runner')
parser.add_argument('instances_file')
parser.add_argument('--correct-switch', action='store_true', default=True, help='guess the correct switch by using double dash')

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)
lines = []
for f in os.listdir(args.input_dir):
    if os.path.isfile(os.path.join(args.input_dir, f)):
        if not f.endswith('.json'):
            raise RuntimeError('file does not end with .json, not sure how to proceed')
    new_args = [
        'python',
        os.path.relpath(os.path.join(args.data_comparison_dir, 'compare.py'), start=args.output_dir),
        os.path.relpath(os.path.join('.', args.input_dir, f), start=args.output_dir),
        os.path.relpath(os.path.join('.', args.output_dir, f[:-5]+'.txt'), start=args.output_dir),
        os.path.relpath(os.path.join('.', args.target_runner), start=args.output_dir),
    ]
    lines.append(shlex.join(new_args))

with open(args.instances_file, encoding='utf-8') as f:
    o = toml.load(f)
    pyrfr_wrapper = o['surrogate_args']['pyrfr_wrapper']
    pyrfr_model = o['surrogate_args']['pyrfr_model']
    config_space = o['surrogate_args']['config_space']
    inst_feat_dict = o['surrogate_args']['inst_feat_dict']
    # convert to relative path
    pyrfr_wrapper = os.path.relpath(os.path.abspath(pyrfr_wrapper), start=args.output_dir)
    pyrfr_model = os.path.relpath(os.path.abspath(pyrfr_model), start=args.output_dir)
    config_space = os.path.relpath(os.path.abspath(config_space), start=args.output_dir)
    inst_feat_dict = os.path.relpath(os.path.abspath(inst_feat_dict), start=args.output_dir)

with open(os.path.join(args.output_dir, 'cmds.sh'), 'w', encoding='utf-8') as f:
    f.write('#!/bin/bash\n')
    f.write('\n')
    model_args = [
        '--pyrfr_wrapper', pyrfr_wrapper,
        '--pyrfr_model', pyrfr_model,
        '--config_space', config_space,
        '--inst_feat_dict', inst_feat_dict,
    ]
    f.write('rm ./nameserver_creds.pkl\n')
    f.write(shlex.join([
        'python',
        '-m', 'epm.webserver.flask_server',
        *model_args,
        '--idle_time', '10000000000', #FIXME: figure out how to disable timeout
        '--pid', '0', # We can also use pid to differenciate but it is not necessary as we use dir.
        '--dir', '.',
        'start',
    ])+' > ./server-log.txt 2>&1 &\n')
    f.write(shlex.join([
        'python',
        os.path.relpath(os.path.join(args.data_comparison_dir, 'wait_for_server.py'), start=args.output_dir),
    ])+' &&\n')
    f.write(' &&\n'.join(lines))
    f.write('\n')
    f.write(shlex.join([
        'python',
        '-m', 'epm.webserver.flask_server',
        *model_args,
        '--pid', '0',
        '--dir', '.',
        'stop',
    ])+'\n')
