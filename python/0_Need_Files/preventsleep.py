import time
import subprocess

# For macOS: use caffeinate to prevent sleep
proc = subprocess.Popen(['caffeinate'])

try:
    while True:
        time.sleep(60)  # Sleep in a loop, minimal CPU usage
except KeyboardInterrupt:
    proc.terminate()