
# %%
import time
import numpy.random as npr
from multiprocessing import Process, Manager

def f(i, l):

    while i[0]<20:
        l.append(npr.random())
        i[0] += 1
        # time.sleep(0.5)

if __name__ == '__main__':
    with Manager() as manager:
        i = manager.list([1])
        l = manager.list([])

        p = Process(target=f, args=(i, l))
        p.start()
        # time.sleep(2)
        p.join()

        print(i)
        print(len(l))
