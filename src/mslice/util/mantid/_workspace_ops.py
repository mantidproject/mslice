import operator
import numpy as np
from uuid import uuid4
from mantid.api._workspaceops import _do_binary_operation
from mantid.kernel.funcinspect import lhs_info
from mslice.workspace.workspace import WorkspaceOperatorMixin
from mslice.workspace.helperfunctions import WorkspaceNameHandler

_binary_operator_map = {
    "Plus": operator.add,
    "Minus": operator.sub,
    "Multiply": operator.mul,
    "Divide": operator.truediv,
    "LessThan": operator.lt,
    "GreaterThan": operator.gt,
    "Or": operator.or_,
    "And": operator.and_,
    "Xor": operator.xor
}


def _binary_op(self, other, algorithm, result_info, inplace, reverse):
    """
    Delegate binary operations (+,-,*,/) performed on a workspace depending on type.

    Unwrap mantid workspace (_raw_ws), perform operation, then rewrap.

    Behaviour depends on type of other:
    List - convert to numpy array, then pass to _binary_op_array method.
    Numpy array - pass to _binary_op_array method.
    Workspace wrapper (of the same type and dimensionality) - unwrap, apply operator to each bin, rewrap.
    Else - try to apply operator to self._raw_ws. Works for numbers, unwrapped workspaces...

    :param other: the rhs (MSlice-wrapped) workspace
    :param algorithm: A string containing the Mantid binary operator algorithm name (e.g. "Plus")
    :param result_info: A tuple containing details of the lhs of the assignment, i.e a = b + c, result_info = (1, 'a')
    :param inplace: True if the operation should be performed inplace
    :param reverse: True if the reverse operator was called, i.e. 3 + a calls __radd__
    :return: new workspace wrapper with same type as self.
    """
    if isinstance(other, list):
        other = np.asarray(other)
    if isinstance(other, self.__class__):
        if _check_dimensions(self, other):
            inner_res = _do_binary_operation(algorithm, self._raw_ws, other._raw_ws, result_info, inplace, reverse)
        else:
            raise RuntimeError("workspaces must have same dimensionality for binary operations (+, -, *, /)")
    elif isinstance(other, np.ndarray):
        inner_res = self._binary_op_array(_binary_operator_map[algorithm], other)
    else:
        inner_res = _do_binary_operation(algorithm, self._raw_ws, other, result_info, inplace, reverse)
    return self.rewrap(inner_res)


def _check_dimensions(self, workspace_to_check):
    """check if a workspace has the same number of bins as self for each dimension"""
    for i in range(self._raw_ws.getNumDims()):
        if self._raw_ws.getDimension(i).getNBins() != workspace_to_check._raw_ws.getDimension(i).getNBins():
            return False
    return True


def _attach_binary_operators():
    def add_operator_func(attr, algorithm, inplace, reverse):
        def op_wrapper(self, other):
            result_info = lhs_info()
            # Replace output workspace name with a unique temporary name hidden in ADS
            if result_info[0] > 0:
                temp_name = WorkspaceNameHandler(str(uuid4())[:8]).get_name(hide_from_ADS=True, mslice_signature=True, temporary_signature=True)
                result_info = (result_info[0], (temp_name,) + result_info[1][1:])
            return _binary_op(self, other, algorithm, result_info, inplace, reverse)

        op_wrapper.__name__ = attr
        setattr(WorkspaceOperatorMixin, attr, op_wrapper)

    operations = {
        "Plus": ("__add__", "__radd__", "__iadd__"),
        "Minus": ("__sub__", "__rsub__", "__isub__"),
        "Multiply": ("__mul__", "__rmul__", "__imul__"),
        "Divide": ("__truediv__", "__rtruediv__", "__itruediv__"),
        "LessThan": "__lt__",
        "GreaterThan": "__gt__",
        "Or": "__or__",
        "And": "__and__",
        "Xor": "__xor__"
    }

    for alg, attributes in operations.items():
        if isinstance(attributes, str):
            attributes = [attributes]
        for attr in attributes:
            add_operator_func(attr, alg, attr.startswith('__i'), attr.startswith('__r'))
