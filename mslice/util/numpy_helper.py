import numpy as np
import warnings


def apply_with_corrected_shape(method, shape_array, array, exception):
    """
    performs an operation on two arrays even if their shapes don't match numpy's broadcasting rules.
    :param method: operation to perform on the two arrays
    :param shape_array: array that has the desired shape
    :param array: array that has at least one axis of the same size as shape_array
    :param exception: exception to raise if array does not have at least one axis of the same size as shape_array
    """
    try:
        return_value = method(shape_array, array)
    except ValueError:
        if array.ndim == 1:  # make 2D
            array = array[:, np.newaxis]
        if array.shape[1] == 1:
            array = np.transpose(array)
        if shape_array.shape[0] in array.shape and shape_array.shape[0] != 1:  # array matches the wrong (leftmost) axis
            return apply_with_swapped_axes(method, shape_array, array)
        elif shape_array.shape[1] == array.shape[1]:
            return method(shape_array, array)
        else:
            raise exception
    return return_value


def apply_with_swapped_axes(method, array, *args):
    out = method(np.transpose(array), *args)
    return np.transpose(out)


def transform_array_to_workspace(array, workspace):
    try:
        if hasattr(workspace, 'getSignalArray'):
            return array.reshape(workspace.getSignalArray().shape)
        elif hasattr(workspace, 'extractY'):
            return array.reshape(workspace.extractY().shape)
        else:
            raise RuntimeError("Unable to extract array from workspace")
    except ValueError:
        raise RuntimeError("Unable to transform array to workspace")


def clean_array(arr):
    arr_clean = []
    for elm in arr:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clean_elm = float(elm)
            if not np.isnan(clean_elm):
                arr_clean.append(clean_elm)
        except ValueError:
            pass
    return arr_clean
