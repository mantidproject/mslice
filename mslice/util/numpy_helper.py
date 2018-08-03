import numpy as np

def apply_with_corrected_shape(method, shape_array, array):
    '''
    performs an operation on two arrays even if shapes don't match numpy's broadcasting rules.
    :param method: operation to perform on the two arrays
    :param shape_array: array that has the desired shape
    :param array: array that has at least one axis of the same size as shape_array
    '''
    if array.ndim == 1: # make 2D
        array = array[:, np.newaxis]
    if array.shape[1] == 1:
        array = np.transpose(array)
    if shape_array.shape[0] in array.shape: # array matches the wrong (leftmost) axis
        return apply_with_swapped_axes(method, shape_array, array)
    return method(shape_array, array)

def apply_with_swapped_axes(method, array, *args):
    out = method(np.transpose(array), *args)
    return np.transpose(out)
