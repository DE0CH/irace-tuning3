import argparse
import subprocess
import signal
import sys
import shlex
import os
import time 

def main():
    parser = argparse.ArgumentParser() #FIXME: change these to positional arguments.
    parser.add_argument('--pyrfr_wrapper', type=str, help='Path to the pyrfr_wrapper.')
    parser.add_argument('--pyrfr_model', type=str, help='Path to the target_algorithm.')
    parser.add_argument('--config_space', type=str, help='Path to the configuration space in pcs format.')
    parser.add_argument('--inst_feat_dict', type=str, help='Path to the instance feature dictionary.')
    parser.add_argument('--seed', type=int, help='Random seed.') #TODO: Check how to pass the seed, though it might be unnecessary.
    parser.add_argument('--dir', type=str, help='Path for the epm server as a unique identifier for the server. It is also the cwd for irace.', default='.')
    parser.add_argument('irace_options', nargs=argparse.REMAINDER, help='Options passed to irace.')
    args = parser.parse_args()
    model_args = [
        '--pyrfr_wrapper', args.pyrfr_wrapper,
        '--pyrfr_model', args.pyrfr_model,
        '--config_space', args.config_space,
        '--inst_feat_dict', args.inst_feat_dict,
    ]

    server_args = [
        sys.executable, 
        '-m', 'epm.webserver.g_unicorn_app',
        *model_args,
        '--idle_time', '10000000000', #FIXME: figure out how to disable timeout
        '--pid', '0', # We can also use pid to differenciate but it is not necessary as we use dir.
        '--dir', args.dir,
    ]
    server = subprocess.Popen(server_args + ['start'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) #FIXME: It's not idea to swallow stderr but it's printing a lot of logs to stderr. Should make it print to stdout.
    def wait_to_boot():
        while True:
            try:
                line = next(server.stderr)
                line = line.decode("utf-8")
                if 'Booting worker with pid' in line:
                    return
            except StopIteration:
                raise RuntimeError('Server unexpectedly stopped.')
    wait_to_boot()
    irace_args = [
        os.path.join(subprocess.check_output(['Rscript', '-e', "cat(system.file(package=\'irace\', \'bin\', mustWork=TRUE))"]).decode('utf-8'), 'irace'),
        *args.irace_options[1:]
    ]
    irace = subprocess.Popen(irace_args, cwd=args.dir) #TODO: capture and log the data.
    irace.wait()
    stopper_args = [
        sys.executable,
        '-m', 'epm.webserver.flask_server',
        *model_args,
        '--pid', '0',
        '--dir', args.dir,
        'stop'
    ]
    try:
        if irace.returncode != 0:
            raise RuntimeError('irace failed.')
    finally:
        cleanup(stopper_args, server)

def cleanup(stopper_args, server):
    stopper = subprocess.run(stopper_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stopper.check_returncode()
    except:
        server.kill() # The stopper failed so we need to kill the server.
        raise RuntimeError('Failed to stop the server.')
    finally:
        server.wait() # Make sure the server is not orphaned.
        if server.returncode != 0:
            raise RuntimeError('Server failed.')


if __name__ == '__main__':
    exit(main())