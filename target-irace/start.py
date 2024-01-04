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
        '-u',
        '-m', 'epm.webserver.flask_server',
        *model_args,
        '--idle_time', '10000000000', #FIXME: figure out how to disable timeout
        '--pid', '0', # We can also use pid to differenciate but it is not necessary as we use dir.
        '--dir', args.dir,
        '--nic_name', 'localhost',
    ]
    try:
        os.remove('./server-stopped.log')
    except OSError:
        pass
    try:
        os.remove('./nameserver_creds.pkl')
    except OSError:
        pass
    if os.fork() == 0:
        os.execv('/bin/sh', ['/bin/sh', '-c', f'{shlex.join(server_args + ["start"])} > ./server-log.log 2>&1 & echo $! > ./server.pid; wait; date > server-stopped.log'])

    has_file = Event()
    event_handler = FileCreateHandler(has_file, 'nameserver_creds.pkl')
    observer = Observer()
    observer.schedule(event_handler, '.', recursive=False)
    observer.start()
    has_file.wait()
    observer.stop()
    observer.join()
    #TODO: DRY, combine this with the one in tools/data_comparison/wait_for_server.py
    while True:
        try:
            with open('nameserver_creds.pkl', 'rb') as f:
                ip, port, _ = pickle.load(f)
            response = requests.get(f'http://{ip}:{port}/status')
            if response.status_code == 200:
                break
        except:
            pass
        finally:
            time.sleep(0.1)
        if os.path.exists('./server-stopped.log'):
            raise RuntimeError('Server stopped unexpectedly.')

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
    if irace.returncode != 0:
        raise RuntimeError('irace failed.')

if __name__ == '__main__':
    try:
        os.remove('./server.pid')
    except OSError:
        pass
    try:
        main()
    finally:
        if os.path.exists('./server-stopped.log'):
            subprocess.run(['/bin/sh', '-c', 'kill -15 $(cat ./server.pid)'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
