"""
multi-process communication based Pipe with python
"""

import os
import time
from multiprocessing import Process, Pipe


def sub_proc(x, name, pipe):
    _in_port, _out_port = pipe

    while True:
        try:
            msg = _in_port.recv()
            print("subp: {} recieve {}".format(name, msg))
        except EOFError:
            print("pip port is closed.")
            break


def sub_proc2(x, name, pipe):
    _in_port, _out_port = pipe

    for i in range(100):
        _out_port.send(x + i)
    _out_port.close()


if __name__ == "__main__":
    in_port, out_port = Pipe()
    subp = Process(target=sub_proc, args=(100, "p1", (in_port, out_port)))
    subp2 = Process(target=sub_proc2, args=(10, "p2", (in_port, out_port)))
    subp2.start()
    subp.start()

    subp.join()
    subp2.join()
    print("Main Process Done.")
