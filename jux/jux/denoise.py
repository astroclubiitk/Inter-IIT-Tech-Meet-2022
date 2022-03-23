import numpy as np
import scipy
import pandas as pd
import plotly.express as px


def smoothen_fft_(lc, thresh=200):
    # a low pass filter
    lc_fft = np.fft.fft(lc)
    lc_fft[thresh : len(lc) - thresh] = 0
    lc_smooth = np.abs(np.fft.ifft(lc_fft))
    return lc_smooth


def smoothening_ma(__x, __y, window_sz, shift):
    """
    Using moving average to smoothen data and linearly interpolating back to original size
    """

    new_norm = []
    new_norm_data_points = []
    # for first frame
    new_norm.append(np.mean(__y[0:window_sz]))
    new_norm_data_points.append(__x[0])
    for i in range(window_sz, len(__y), shift):
        tmp = np.mean(__y[i : i + shift])
        new_norm.append(tmp)
        new_norm_data_points.append(__x[i])
    new_norm = np.array(new_norm)
    new_norm_data_points = np.array(new_norm_data_points)
    xnew = np.linspace(__x[0], __x[0] + len(__x), __x[0] + len(__x))

    # interpolating back to original size
    f = scipy.interpolate.interp1d(
        new_norm_data_points, new_norm, fill_value="extrapolate", kind="linear"
    )
    ynew = f(xnew)
    return xnew, ynew


def smoothening_fft(lc, thresh=200, should_plot=False):
    lc_fft = np.fft.fft(lc)
    lc_fft[thresh : len(lc) - thresh] = 0
    lc_smooth = np.abs(np.fft.ifft(lc_fft)) + 1e-5
    if should_plot:
        px.line(pd.DataFrame(lc_smooth))
    xnew = np.linspace(0, len(lc), len(lc))
    return xnew, abs(lc_smooth)
