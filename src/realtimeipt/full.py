
from threading import Thread
from time import sleep
import sys

from realtimeipt.api import SERVE
from realtimeipt.roi import main as roi

def server_monitor(proc):
    while proc.poll() is None:
        sleep(1)
    sys.exit()

def main():
    import subprocess as sp
    from shlex import split
    server = sp.Popen(split(SERVE))
    Thread(target=lambda: server_monitor(server)).start()
    roi(*sys.argv[1:])

