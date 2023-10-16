from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from watchdog.events import FileCreatedEvent
import os
import pickle
import requests
from threading import Event
import time

class FileCreateHandler(FileSystemEventHandler):
    def __init__(self, flag, file_name):
        self.flag = flag
        self.file_name = file_name
    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and os.path.basename(os.path.normpath(event.src_path)) == self.file_name:
            self.flag.set()

has_file = Event()
event_handler = FileCreateHandler(has_file, 'nameserver_creds.pkl')
observer = Observer()
observer.schedule(event_handler, '.', recursive=False)
observer.start()
has_file.wait()
observer.stop()
observer.join()
while True: #FIXME: We need to poll the server to make sure it is running. This is not ideal. This is because nameserver_creds.pkl is created before the server is actually running. I haven't figured out a way to run a command after everything is ready.
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
