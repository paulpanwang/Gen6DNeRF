import time
from colorama import Fore, Back, Style


def get_time_second(f):
    """_summary_

    Args:
        second (time): return function cost time
    """
    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print( '{} cost: {} s'.format(f.__name__, e_time - s_time))
        return res
    return inner

def get_time_millisecond(f):
    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('{} cost: {} ms'.format(f.__name__, 1000*(e_time - s_time)) )
        return res
    return inner