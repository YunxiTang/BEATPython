from multiprocessing import Process, Queue
import os
import queue
import time


class publisher(Process):
    def __init__(self, name, queue: Queue, **kargs):
        super().__init__(**kargs)
        self._name = name
        self._queue = queue
        self._count = 0

    def run(self):
        while True:
            msg = self._count
            try:
                self._queue.put_nowait(msg)
                print("{} pub {}".format(self._name, msg))
            except queue.Full:
                pass
            self._count += 1
            time.sleep(0.001)


class subscriber(Process):
    def __init__(self, name, queue: Queue, **kargs):
        super().__init__(**kargs)
        self._name = name
        self._queue = queue
        self._pre_msg = None

    def run(self):
        while True:
            if not self._queue.empty():
                qsize = self._queue.qsize()
                msg = self._queue.get()
                print("{} get {} with qsize: {}".format(self._name, msg, qsize))
                print("=========================")
                self._pre_msg = msg
            else:
                print(
                    "empty queue, use last msg. {} get {}".format(
                        self._name, self._pre_msg
                    )
                )
            time.sleep(0.02)


if __name__ == "__main__":
    state_queue = Queue(maxsize=1)
    ctrl_queue = Queue(maxsize=100)

    pub = publisher("pub", state_queue)
    sub = subscriber("sub", state_queue)

    pub.start()
    sub.start()

    pub.join()
    sub.join()
