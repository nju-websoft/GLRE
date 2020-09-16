#!/usr/bin/python3

import torch
import six
import numpy


def to_device(device, x):
    if device is None:
        return torch.as_tensor(x)
    return torch.as_tensor(x).to(device)


def concat_examples(batch, device=None, padding=-1):
    assert device is None or isinstance(device, torch.device)
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    if isinstance(first_elem, tuple):
        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            result.append(to_device(device, _concat_arrays(
                [example[i] for example in batch], padding[i])))

        return tuple(result)

    elif isinstance(first_elem, dict):
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}
        padding['multi_rels'] = 0
        padding['entA_mapping'] = 0
        padding['entB_mapping'] = 0
        padding['dep_adj'] = 0

        for key in first_elem:
            # flag = True
            # if key == "ent_sen_mask":
            #     flag = False
            result[key] = to_device(device, _concat_arrays(
                [example[key] for example in batch], padding[key]))

        return result

    else:
        return to_device(device, _concat_arrays(batch, padding))


def _concat_arrays(arrays, padding):
    # Convert `arrays` to numpy.ndarray if `arrays` consists of the built-in
    # types such as int, float or list.
    if not isinstance(arrays[0], type(torch.get_default_dtype())):
        arrays = numpy.asarray(arrays)

    if padding is not None:
        arr_concat = _concat_arrays_with_padding(arrays, padding)
    else:
        arr_concat = numpy.concatenate([array[None] for array in arrays])

    return arr_concat


def _concat_arrays_with_padding(arrays, padding):
    shape = numpy.array(arrays[0].shape, dtype=int)
    for array in arrays[1:]:
        if numpy.any(shape != array.shape):
            numpy.maximum(shape, array.shape, shape)
    shape = tuple(numpy.insert(shape, 0, len(arrays)))

    result = numpy.full(shape, padding, dtype=arrays[0].dtype)
    for i in six.moves.range(len(arrays)):
        src = arrays[i]
        slices = tuple(slice(dim) for dim in src.shape)
        result[(i,) + slices] = src

    return result
