import argparse
import subprocess
import signal
import sys
import shlex
import os
import time 
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from watchdog.events import FileCreatedEvent
from threading import Event
import pickle
import requests

class FileCreateHandler(FileSystemEventHandler):
    def __init__(self, flag, file_name):
        self.flag = flag
        self.file_name = file_name
    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and os.path.basename(os.path.normpath(event.src_path)) == self.file_name:
            self.flag.set()

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
    try:
        os.remove('./nameserver_creds.pkl')
    except OSError:
        pass 
    if os.fork() == 0: #FIXME: Using fork is not ideal, best to use subprocess, but there's a bug when using subprocess. It just stops abitrairly.
        os.execv('/bin/sh', ['/bin/sh', '-c', f'{shlex.join(server_args + ["start"])} > ./server-log.txt 2>&1']) #FIXME: I don't like using shell.

    has_file = Event()
    event_handler = FileCreateHandler(has_file, 'nameserver_creds.pkl')
    observer = Observer()
    observer.schedule(event_handler, '.', recursive=False)
    observer.start()
    has_file.wait()
    observer.stop()
    observer.join()
    with open('nameserver_creds.pkl', 'rb') as f:
        ip, port, _ = pickle.load(f)
    while True: #FIXME: We need to poll the server to make sure it is running. This is not ideal. This is because nameserver_creds.pkl is created before the server is actually running. I haven't figured out a way to run a command after everything is ready.
        try:
            response = requests.get(f'http://{ip}:{port}/status')
            if response.status_code == 200:
                break
        except:
            pass
        finally:
            time.sleep(0.1)

    irace_args = [
        os.path.join(subprocess.check_output(['Rscript', '-e', "cat(system.file(package=\'irace\', \'bin\', mustWork=TRUE))"]).decode('utf-8'), 'irace'),
        *args.irace_options[1:]
    ]
    
    irace = subprocess.run(irace_args, cwd=args.dir) #TODO: capture and log the data. For some reason if I try to capture the data here the flask_worker just says ERROR:EPM Worker:Server is not running on 127.0.0.1:41391

    stopper_args = [
        sys.executable,
        '-m', 'epm.webserver.flask_server',
        *model_args,
        '--pid', '0',
        '--dir', args.dir,
        'stop'
    ]
    stopper = subprocess.run(stopper_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if irace.returncode != 0:
        raise RuntimeError('irace failed.')

if __name__ == '__main__':
    exit(main())