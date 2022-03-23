import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .helper import exp_fit_func, inverse_exp_func, exp_func


def exp_curve_fit_(x_range, ln_y_range):
    popc, pcov = curve_fit(exp_fit_func, x_range, ln_y_range)
    ln_a, b = popc
    a = np.exp(ln_a)
    return a, b


def get_interm_zip_features_(ynew, _s4, _p4, _e1):
    start_times = []
    peak_times = []
    end_times = []
    peak_intensities = []
    for i in range(len(_s4)):
        if (_p4[i] - _s4[i] > 0) and (_e1[i] - _p4[i] > 0):
            start_times.append(_s4[i])
            peak_times.append(_p4[i])
            end_times.append(_e1[i])
            peak_intensities.append(ynew[_p4[i]])
    return start_times, peak_times, end_times, peak_intensities


def get_interm_zip_(h1, h2, h3, h4):
    _zip = pd.DataFrame(zip(h1, h2, h3, h4))
    _zip.columns = ["start_time", "peak_time", "end_time", "peak_intensity"]
    return _zip


def get_final_zip_features(xnew, ynew, _zip):
    st = _zip["start_time"]
    pt = _zip["peak_time"]
    et = _zip["end_time"]
    pi = _zip["peak_intensity"]
    y_min = np.min(ynew)
    final_st = []
    final_pt = []
    final_et = []
    est_et = []
    final_si = []
    final_pi = []
    final_err = []
    final_bc = []
    _class = []
    for i in range(len(st)):
        x_range = [int(xnew[j] - xnew[pt[i]]) for j in range(pt[i], et[i])]
        ln_y_range = [np.log(ynew[j]) for j in range(pt[i], et[i])]
        try:
            popc, pcov = curve_fit(exp_fit_func, x_range, ln_y_range)
            ln_a, b = popc
            a = np.exp(ln_a)
            # the 7th filter, can't allow increasing exponential so-called-flares!
            # _calc_et is estimated end time from the analytical function fitted
            if b < 0:
                continue
            _calc_et = inverse_exp_func(ynew[st[i]], a, b)
            final_st.append(st[i])
            final_pt.append(pt[i])
            final_et.append(et[i])
            final_pi.append(pi[i])
            final_si.append(ynew[st[i]])
            est_et.append(_calc_et + pt[i])
            final_bc.append((ynew[st[i]] + ynew[et[i]]) / 2)
            y_dash = []
            y_diff = []
            y_proj = []
            x_proj = []
            for _i, j in enumerate(x_range):
                __y = exp_func(xnew[j], a, b)
                y_dash.append(__y)
                y_diff.append(abs(np.exp(ln_y_range[_i]) - __y))
            for j in range(et[i] - pt[i], _calc_et):
                if (j + pt[i]) < len(xnew):
                    x_proj.append(xnew[j + pt[i]])
                    y_proj.append(exp_func(xnew[j], a, b))
            # error is sum(difference between fitted and actual) / ((peak intensity - minimum intensity) * duration from peak to actual end)
            final_err.append((np.sum(y_dash)) / ((pi[i] - y_min) * (len(x_range))))
            val = np.log10(pi[i] / 25)
            _str = ""
            _val = str(int(val * 100) / 10)[-3:]
            if int(val) < 1:
                _str = "A" + _val
            elif int(val) == 1:
                _str = "B" + _val
            elif int(val) == 2:
                _str = "C" + _val
            elif int(val) == 3:
                _str = "M" + _val
            elif int(val) > 3:
                _str = "X" + _val
            _class.append(_str)
        except:
            print("Error in curve fitting")
    return (
        final_st,
        final_pt,
        final_et,
        est_et,
        final_si,
        final_pi,
        final_bc,
        final_err,
        _class,
    )


def get_final_zip(g1, g2, g3, g4, g5, g6, g7, g8, g9):
    final_zip = pd.DataFrame(zip(g1, g2, g3, g4, g5, g6, g7, g8, g9))
    final_zip.columns = [
        "start_time",
        "peak_time",
        "end_time",
        "est_end_time",
        "start_intensity",
        "peak_intensity",
        "background_counts",
        "error",
        "class",
    ]
    return final_zip
