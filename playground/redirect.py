import os
import sys
sys.stdout = open('file', 'w')
if os.fork() == 0:
    print("hello world")
    os.execvp('echo', ['echo', 'hello'])