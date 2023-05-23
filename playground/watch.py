from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from watchdog.events import FileCreatedEvent
from threading import Event
import os


has_file = Event()
class FileCreateHandler(FileSystemEventHandler):
    def __init__(self, flag, file_name):
        self.flag = flag
        self.file_name = file_name
    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and os.path.basename(os.path.normpath(event.src_path)) == self.file_name:
            self.flag.set()

event_handler = FileCreateHandler(has_file, 'nameserver_creds.pkl')
observer = Observer()
observer.schedule(event_handler, '.', recursive=False)
observer.start()
has_file.wait()
observer.stop()
observer.join()