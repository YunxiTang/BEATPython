'''
    multi-process with python
'''
import os
import time
from multiprocessing import Process, Pipe


def run_proc1(name):
    for i in range(10):
        print('[{1}]: Run child process {0} at {2}'.format(name, os.getpid(), i))
        time.sleep(1)

def run_proc2(name):
    for i in range(20):
        print('[{1}]: Run child process {0} at {2}'.format(name, os.getpid(), i))
        time.sleep(2)


class MyProcess(Process):
    def __init__(self, interval: float, **kargs):
        super().__init__(**kargs)
        self.interval = interval

    def run(self):
        print('subprocess pid: {} || parent pid: {}'.format(os.getpid(), os.getppid()))
        t_start = time.time()
        time.sleep(self.interval)
        t_end = time.time()
        print('Used time: {}'.format(t_end - t_start))
        return super().run()


if __name__ == '__main__':
    print('Parent Process pid {}'.format(os.getpid()))
    p1 = Process(
        target=run_proc1,
        args=('p1',)
    )
    p2 = Process(
        target=run_proc2,
        args=('p2',)
    )
    p3 = MyProcess(2, name='p3')
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    print('Done.')
    