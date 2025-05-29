from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def cal_elasped_time(elapsed_time):
    min, sec = divmod(elapsed_time, 60)
    if min > 60:
        hour, min = divmod(min, 60)
        time_ = f'{hour} h, {round(min, 0)} min, {round(sec, 0)} s'
    else:
        time_ = f'{round(min, 0)} min, {round(sec, 0)} s'
    return time_
