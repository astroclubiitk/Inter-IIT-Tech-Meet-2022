import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pandas as pd
import scipy.interpolate
import plotly.express as px
from math import sqrt
import os
from scipy.optimize import curve_fit
from numpy import exp, log

""""File Paths"""
def return_folder_paths(file_name):
    raw_files_path = f'./{file_name}/xsm/data/{file_name[8:12]}/{file_name[12:14]}/{file_name[14:16]}/raw'
    calibrated_files_path = f'./{file_name}/xsm/data/{file_name[8:12]}/{file_name[12:14]}/{file_name[14:16]}/calibrated'
    return raw_files_path, calibrated_files_path

"""Returning a Light Curve from a file"""
def lightcurve(file, should_plot=False):
    t = Table.read(file)
    tmp = t["RATE"]
    if should_plot:
        plt.figure(figsize=(25, 7))
        plt.scatter(range(len(tmp)), tmp, s=1)
        plt.show()
    return tmp

"""Plotting the curve using Plotly"""
def plot_as_plotly(_x, _y, _title):
    _df = pd.DataFrame(zip(_x, _y))
    _df.columns = ['X-Axis', 'Y-Axis']
    _fig = px.line(_df, x='X-Axis', y='Y-Axis', title=_title)
    _fig.show()
    return

"""Return the slope calculated between two data points in 2-Dimensional space"""
def get_slope(x1, x2, y1, y2):
    return ((y1-y2) / (x1-x2))

"""Return the Eucleidean distance between two points"""
def pythagorean(x1, x2, y1, y2):
    _y = (y1-y2)*(y1-y2)
    _x = (x1-x2)*(x1-x2)
    return sqrt(_x + _y)

"""Helper functions to fit on the detected flares"""
k = 0.5

def exp_fit_func(x, ln_a, b):
    t = (x ** k)
    return (ln_a - b*t)

def exp_func(x, a, b):
    t = -1 * b * (x ** k)
    return (a * np.exp(t))

def inverse_exp_func(y, a, b):
    t1 = log(y) - log(a)
    t2 = -1 * t1 / b
    return int(t2 ** (1. /k))

def smoothening(rand_lc, window_sz):
    new_norm = []
    new_norm_data_points = []
    for i in range(1, len(rand_lc), window_sz):
        tmp = np.mean(rand_lc[i:i+window_sz])
        new_norm.append(tmp)
        new_norm_data_points.append(i)
    new_norm = np.array(new_norm)
    new_norm_data_points = np.array(new_norm_data_points)
    tck, u = scipy.interpolate.splprep([new_norm_data_points, new_norm], s=0)
    xnew, ynew = scipy.interpolate.splev(np.linspace(0, 1, len(rand_lc)), tck, der=0)
    return xnew, ynew

"""Getting extremas functions"""
def get_lvl_0_extremas(xnew, ynew, should_plot=False):
    _s0 = []
    _p0 = []
    for i in range(1, len(ynew)-1):
        if (ynew[i]>ynew[i-1]) and (ynew[i]>ynew[i+1]):
            _p0.append(i)
        elif (ynew[i]<ynew[i-1]) and (ynew[i]<ynew[i+1]):
            _s0.append(i)
    if should_plot:
        plt.figure()
        plt.title("Level 0 Maximas")
        plt.plot([xnew[i] for i in _p0], [ynew[i] for i in _p0], 'o', xnew, ynew)
        plt.show()
        plt.figure()
        plt.title("Level 0 Minimas")
        plt.plot([xnew[i] for i in _s0], [ynew[i] for i in _s0], 'o', xnew, ynew)
        plt.show()
    return _s0, _p0

def get_lvl_1_extremas(xnew, ynew, _s0, _p0, should_plot=False):
    _p1 = []
    _s1 = []
    l = len(_p0)
    is_increasing = True
    for i in range(l-1):
        if is_increasing:
            while (i < l-1) and (ynew[_p0[i]] < ynew[_p0[i+1]]):
                i += 1
            _p1.append(_p0[i])
            is_increasing = False
        else:
            if (i < l-1):
                if (ynew[_p0[i]] < ynew[_p0[i+1]]):
                    is_increasing = True
    for i in range(1, len(_p1)):
        k = 0
        for j in range(len(_s0)):
            if (xnew[_s0[j]] > xnew[_p1[i]]):
                k = j
                break
        tmp_sl = get_slope(xnew[_s0[k]], xnew[_p1[i]], ynew[_s0[k]], ynew[_p1[i]])
        tmp_val = _s0[k]
        while (xnew[_s0[k]] > xnew[_p1[i-1]]):
            t_sl = get_slope(xnew[_s0[k]], xnew[_p1[i]], ynew[_s0[k]], ynew[_p1[i]])
            if (t_sl > tmp_sl):
                tmp_sl = t_sl
                tmp_val = _s0[k]
            k -= 1
            if k<0:
                break
        _s1.append(tmp_val)
    if should_plot:
        plt.figure()
        plt.title("Level 1 Maximas")
        plt.plot([xnew[i] for i in _p1], [ynew[i] for i in _p1], 'o', xnew, ynew)
        plt.show()
        plt.figure()
        plt.title("Level 1 Minimas")
        plt.plot([xnew[i] for i in _s1], [ynew[i] for i in _s1], 'o', xnew, ynew)
        plt.show()
    return _s1, _p1

def get_lvl_2_extremas(xnew, ynew, _s1, _p1, should_plot=False):
    _s2 = []
    _p2 = []
    for i in range(len(_p1)):
        for j in range(len(_s1)-1):
            if (xnew[_s1[j+1]] > xnew[_p1[i]]):
                _s2.append(_s1[j])
                _p2.append(_p1[i])
                break
    if should_plot:
        plt.figure()
        plt.title("Level 2 Maximas")
        plt.plot([xnew[i] for i in _p2], [ynew[i] for i in _p2], 'o', xnew, ynew)
        plt.show()
        plt.figure()
        plt.title("Level 2 Minimas")
        plt.plot([xnew[i] for i in _s2], [ynew[i] for i in _s2], 'o', xnew, ynew)
        plt.show()
    return _s2, _p2

def get_lvl_3_extremas(xnew, ynew, _s2, _p2, should_plot=False):
    _s3 = []
    _p3 = []
    s_x = np.mean(np.array([xnew[_p2[i]] for i in range(len(_p2))]))
    s_y = np.mean(np.array([ynew[_p2[i]] for i in range(len(_p2))]))
    s_y_dash = np.std(np.array([ynew[_p2[i]] for i in range(len(_p2))]))
    threshold_time = s_x * 0.004
    threshold_height = s_y * 0.15
    for i in range(len(_p2)):
        y__ = ynew[_p2[i]]-ynew[_s2[i]]
        if (y__ > threshold_height):
            if (xnew[_p2[i]]-xnew[_s2[i]] > threshold_time) or (y__ > s_y):
                if (ynew[_p2[i]] > s_y_dash):
                    _s3.append(_s2[i])
                    _p3.append(_p2[i])
    if should_plot:
        plt.figure()
        plt.title("Level 3 Maximas")
        plt.plot([xnew[i] for i in _p3], [ynew[i] for i in _p3], 'o', xnew, ynew)
        plt.show()
        plt.figure()
        plt.title("Level 3 Minimas")
        plt.plot([xnew[i] for i in _s3], [ynew[i] for i in _s3], 'o', xnew, ynew)
        plt.show()
    return _s3, _p3

def get_lvl_4_extremas(xnew, ynew, _s3, _p3, should_plot=False):
    _s4 = []
    _p4 = []
    s = set()
    for i in range(len(_s3)):
        if _s3[i] in s:
            continue
        s.add(_s3[i])
        _s4.append(_s3[i])
        _p4.append(_p3[i])
    if should_plot:
        plt.figure()
        plt.title("Final Start and Peak")
        plt.plot([xnew[i] for i in _p4], [ynew[i] for i in _p4], 'o', [xnew[i] for i in _s4], [ynew[i] for i in _s4], 'x', xnew, ynew)
        plt.show()
    return _s4, _p4

def get_lvl_0_ends(xnew, ynew, _s4, _p4, _s0, should_plot=False):
    _e0 = []
    for i in range(len(_p4)):
        for j in range(_p4[i], len(xnew)):
            if i == len(_p4) - 1:
                if (xnew[_s0[-1]] < xnew[_p4[i]]):
                    _e0.append(len(xnew)-1)
                    break
                elif j == len(xnew) - 1:
                    _e0.append(len(xnew)-1)
            if (ynew[j] < (ynew[_p4[i]] + ynew[_s4[i]])/2):
                _e0.append(j)
                break
            if i+1<len(_s4):
                if (xnew[j] > xnew[_s4[i+1]]):
                    _e0.append(j-1)
                    break
    if should_plot:
        plt.figure()
        plt.title("Level 0 Ends")
        plt.plot([xnew[i] for i in _p4], [ynew[i] for i in _p4], 'o', [xnew[i] for i in _e0], [ynew[i] for i in _e0], 'x', xnew, ynew)
        plt.show()
    return _e0

def get_lvl_1_ends(xnew, ynew, _s0, _p4, _e0, should_plot=False):
    _e1 = []
    for i in range(len(_e0)):
        if (ynew[_e0[i]] < ynew[_p4[i]]):
            _e1.append(_e0[i])
        else:
            for j in range(len(_s0)):
                if (xnew[_s0[j]] > xnew[_p4[i]]):
                    if (ynew[_s0[j+1]] > ynew[_s0[j]]):
                        _e1.append(_s0[j])
                        break
    if should_plot:
        plt.figure()
        plt.title("Peaks and Ends")
        plt.plot([xnew[i] for i in _p4], [ynew[i] for i in _p4], 'o', [xnew[i] for i in _e1], [ynew[i] for i in _e1], 'x', xnew, ynew)
        plt.show()
    return _e1

def get_interm_zip(ynew, _s4, _p4, _e1):
    start_times = []
    peak_times = []
    end_times = []
    peak_intensities = []
    for i in range(len(_s4)):
        if (_p4[i]-_s4[i] > 0) and (_e1[i]-_p4[i] > 0):
            start_times.append(_s4[i])
            peak_times.append(_p4[i])
            end_times.append(_e1[i])
            peak_intensities.append(ynew[_p4[i]])
    _zip = pd.DataFrame(zip(start_times, peak_times, end_times, peak_intensities))
    _zip.columns = ['start_time', 'peak_time', 'end_time', 'peak_intensity']
    return _zip

def get_final_zip(xnew, ynew, _zip):
    st = _zip['start_time']
    pt = _zip['peak_time']
    et =  _zip['end_time']
    pi = _zip['peak_intensity']
    y_min = np.min(ynew)
    final_st = []
    final_pt = []
    final_et = []
    est_et = []
    final_si = []
    final_pi = []
    final_err = []
    final_bc = []
    for i in range(len(st)):
        x_range = [int(xnew[j]-xnew[pt[i]]) for j in range(pt[i], et[i])]
        ln_y_range = [np.log(ynew[j]) for j in range(pt[i], et[i])]
        popc, pcov = curve_fit(exp_fit_func, x_range, ln_y_range)
        ln_a, b = popc
        a = np.exp(ln_a)
        if (b < 0):
            continue
        _calc_et = inverse_exp_func(ynew[st[i]], a, b)
        final_st.append(st[i])
        final_pt.append(pt[i])
        final_et.append(et[i])
        final_pi.append(pi[i])
        final_si.append(ynew[st[i]])
        est_et.append(_calc_et + pt[i])
        final_bc.append((ynew[st[i]]+ynew[et[i]])/2)
        y_dash = []
        y_diff = []
        y_proj = []
        x_proj = []
        for _i, j in enumerate(x_range):
            __y = exp_func(xnew[j], a, b)
            y_dash.append(__y)
            y_diff.append(abs(exp(ln_y_range[_i]) - __y))
        for j in range(et[i]-pt[i], _calc_et):
            if ((j + pt[i]) < len(xnew)):
                x_proj.append(xnew[j + pt[i]])
                y_proj.append(exp_func(xnew[j], a, b))
        _y_ = np.array(y_diff)
        final_err.append((np.sum(_y_)) / ((pi[i] - y_min) * (len(x_range))))
    final_zip = pd.DataFrame(zip(final_st, final_pt, final_et, est_et, final_si, final_pi, final_bc, final_err))
    final_zip.columns = ['start_time', 'peak_time', 'end_time', 'est_end_time', 'start_intensity', 'peak_intensity', 'background_counts', 'error']
    return final_zip

def get_model_features(final_zip):
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    for i in range(len(final_zip)):
        t1 = (final_zip['end_time'][i] - final_zip['start_time'][i]) / (final_zip['peak_time'][i] - final_zip['start_time'][i])
        f1.append(t1)
        t2 = (final_zip['peak_intensity'][i] - final_zip['background_counts'][i]) / (final_zip['background_counts'][i] - final_zip['start_intensity'][i])
        f2.append(t2)
        t3 = (final_zip['peak_intensity'][i] - final_zip['start_intensity'][i]) / (final_zip['peak_time'][i] - final_zip['start_time'][i])
        f3.append(t3)
        t4 = final_zip['error'][i]
        f4.append(t4)
    tmp = pd.DataFrame(zip(f1, f2, f3, f4))
    tmp.columns = ['time_ratio', 'intensity_ratio', 'bandwidth', 'error']
    return tmp


"""Main function"""
def main():
    files = os.listdir()
    flag = True
    valid_dirs = []
    for file in files:
        if file[:3] == 'ch2':
            valid_dirs.append(file)
    for file in valid_dirs:
        raw, calib = return_folder_paths(file)
        path_to_lc = calib + '/' + file + '_level2.lc'
        if os.path.isfile(path_to_lc):
            print("Running on file {}".format(path_to_lc))
            rand_lc = lightcurve(path_to_lc, should_plot=False)
            xnew, ynew = smoothening(rand_lc, 40)    

            _s0, _p0 = get_lvl_0_extremas(xnew, ynew, should_plot=False)
            _s1, _p1 = get_lvl_1_extremas(xnew, ynew, _s0, _p0, should_plot=False)
            _s2, _p2 = get_lvl_2_extremas(xnew, ynew, _s1, _p1, should_plot=False)
            _s3, _p3 = get_lvl_3_extremas(xnew, ynew, _s2, _p2, should_plot=False)
            _s4, _p4 = get_lvl_4_extremas(xnew, ynew, _s3, _p3, should_plot=False)
            
            _e0 = get_lvl_0_ends(xnew, ynew, _s4, _p4, _s0)
            _e1 = get_lvl_1_ends(xnew, ynew, _s0, _p4, _e0)
            if len(_e1) != 0:
                _zip = get_interm_zip(ynew, _s4, _p4, _e1)
                final_zip = get_final_zip(xnew, ynew, _zip)
                model_zip = get_model_features(final_zip)
                with open('resultsdata.txt','a') as f:
                    dfAsString = model_zip.to_string(header=False, index=False)
                    f.write(dfAsString)
                    f.write('\n')

if __name__ == '__main__':
    main()
    