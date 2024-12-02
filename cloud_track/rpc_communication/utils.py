import base64
import io
import pickle

import PIL
import torch

to_serialize = (PIL.Image.Image, torch.Tensor)
has_cuda = (
    torch.cuda.is_available() and False
)  # unpickling with cuda on jetson takes forever -> This is why it's disabled here. TODO: Make this configurable!


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def deserialize_kwargs(kwargs):
    kwargs = {
        key: deserialize(value) if is_pickled(value) else value
        for key, value in kwargs.items()
    }
    return kwargs


def deserialize_args(args):
    args = deserialize_list(args)

    return args


def serialize_kwargs(kwargs):
    kwargs = {
        key: serialize(value) if isinstance(value, to_serialize) else value
        for key, value in kwargs.items()
    }

    return kwargs


def serialize_args(args):
    args = serialize_list(args)
    return args


def serialize_results(results):
    results = serialize_list(results)

    return results


def deserialize_results(results):
    results = deserialize_list(results)

    return results


def serialize_list(lst):
    lst = [
        serialize(elem) if isinstance(elem, to_serialize) else elem
        for elem in lst
    ]

    return lst


def deserialize_list(lst):
    lst = [deserialize(elem) if is_pickled(elem) else elem for elem in lst]

    return lst


def is_pickled(obj):
    return isinstance(obj, dict) and "pickle" in obj.keys()
    # return isinstance(obj, bytes)


def serialize(obj):
    # start timer

    obj = pickle.dumps(obj)

    # create base64 string
    obj = base64.b64encode(obj).decode("utf-8")
    obj = {"pickle": obj}
    return obj


def deserialize(obj):
    obj = obj["pickle"]
    # base64 decode
    obj = base64.b64decode(obj)

    if has_cuda:
        obj = pickle.loads(obj)
    else:
        obj = CPU_Unpickler(io.BytesIO(obj)).load()
    return obj
