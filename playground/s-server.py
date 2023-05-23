#!/usr/bin/python

import socket
import os, os.path
import time
from collections import deque

if os.path.exists("/tmp/socket_test.s"):
  os.remove("/tmp/socket_test.s")

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind("/tmp/socket_test.s")
while True:
  server.listen(1)
  conn, addr = server.accept()
  datagram = conn.recv(1024)
  conn.recv(1024)
  if datagram:
    print(datagram)
    conn.send("-1\n".encode('utf-8'))
    conn.close()
    print('connection closed')