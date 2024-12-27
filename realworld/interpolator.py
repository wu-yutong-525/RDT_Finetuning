import numpy as np
from scipy.interpolate import interp1d

# def interpolate(v1, v2, gap=0.001, length=None):
#     """Perform linear interpolation between v1 and v2.

#     Parameters:
#         v1 (array-like): First value.
#         v2 (array-like): Second value.
#         gap (float): Gap between interpolated values.
#         length (int | None): Length of the interpolated sequence,
#             `None` for whole sequence.

#     Returns:
#         numpy.ndarray: Interpolated values.
#     """
#     try:
#         v1 = np.asarray(v1)
#         v2 = np.asarray(v2)
#         assert v1.shape == v2.shape, "Input values must have the same shape"
#         assert gap > 0, "Gap must be a positive value"
#         diff = np.abs(v1 - v2)
#         max_diff = np.max(diff)
#         steps = max(int(np.ceil(max_diff / gap)), 1)
#         interp = interp1d([0, steps], np.stack([v1, v2]), axis=0, assume_sorted=True)
#         length = min(length or steps, steps)
#         assert length > 0, "Length must be a positive value"
#         return interp(np.arange(1, length + 1, 1))
#     except (ValueError, AssertionError) as e:
#         print("Error:", e)

def interpolate(v1, v2, gap=0.001, length=None):
    """Perform linear interpolation between v1 and v2.

    Parameters:
        v1 (array-like): First value.
        v2 (array-like): Second value.
        gap (float): Gap between interpolated values.
        length (int | None): Length of the interpolated sequence,
            `None` for whole sequence.

    Returns:
        numpy.ndarray: Interpolated values.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    max_diff = np.max(np.abs(v1 - v2))
    steps = max(int(np.ceil(max_diff / gap)), 1)
    length = min(length or steps, steps)
    delta = (v2 - v1) / steps
    start = v1 + delta
    end = v1 + delta * length

    return np.linspace(start, end, length, axis=0)
