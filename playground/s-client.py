#!/usr/bin/python

import socket
import os, os.path
import time
from collections import deque

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.connect("/tmp/socket_test.s")
server.sendall(("Hello World\n"*1).encode("utf-8"))
