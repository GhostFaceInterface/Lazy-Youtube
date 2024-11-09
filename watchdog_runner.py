from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os
import subprocess

class Watcher(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.run_script()

    def on_modified(self, event):
        if event.src_path.endswith("main.py"):
            print("Dosyada değişiklik algılandı, script yeniden başlatılıyor...")
            self.restart_script()

    def run_script(self):
        self.process = subprocess.Popen(["python", "main.py"])

    def restart_script(self):
        if self.process:
            self.process.kill()
            time.sleep(1)
            self.run_script()

if __name__ == "__main__":
    event_handler = Watcher()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
