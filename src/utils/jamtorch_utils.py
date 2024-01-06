import numpy as np
import torch as th
import six
import collections
from jamtorch.data import num_to_groups
from jamtorch.utils import no_grad_func

SKIP_TYPES = six.string_types

def _as_numpy(cur_obj):
    if isinstance(cur_obj, SKIP_TYPES):
        return cur_obj
    if th.is_tensor(cur_obj):
        return cur_obj.cpu().numpy()
    return np.array(cur_obj)

def stmap(func, iterable):
    if isinstance(iterable, six.string_types):
        return func(iterable)
    elif isinstance(iterable, (collections.abc.Sequence, collections.UserList)):
        return [stmap(func, v) for v in iterable]
    elif isinstance(iterable, collections.abc.Set):
        return {stmap(func, v) for v in iterable}
    elif isinstance(iterable, (collections.abc.Mapping, collections.UserDict)):
        return {k: stmap(func, v) for k, v in iterable.items()}
    else:
        return func(iterable)
    
def as_numpy(obj):
    return stmap(_as_numpy, obj)