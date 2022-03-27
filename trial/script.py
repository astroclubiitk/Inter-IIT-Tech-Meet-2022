import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pandas as pd
import scipy.interpolate
import plotly.express as px
from math import sqrt
import os, random
from scipy.optimize import curve_fit
from numpy import exp, log


# """Extracting all files"""
# files = os.listdir(".")
# done_zip_files = []
# for file in files:
#     if file[-3:] == "zip":
#         flag = False
#         with zipfile.ZipFile(file, 'r') as zipref:
#             os.mkdir(file[:-4])
#             zipref.extractall(file[:-4])
#         done_zip_files.append(file)

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
        print(len(tmp))
        plt.figure(figsize=(25, 7))
        plt.scatter(range(len(tmp)), tmp, s=1)
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


"""Main function"""
def main():
    files = os.listdir()
    flag = True
    valid_dirs = []
    for file in files:
        if file[:3] == 'ch2':
            valid_dirs.append(file)
    random.seed(0)
    file = random.choice(valid_dirs)
    raw, calib = return_folder_paths(file)
    path_to_lc = calib + '/' + file + '_level2.lc'
    rand_lc = lightcurve(path_to_lc, should_plot=True)
    print(file)

    window_sz = 40
    new_norm = []
    new_norm_data_points = []
    for i in range(1, len(rand_lc), window_sz):
        tmp = np.mean(rand_lc[i:i+window_sz])
        new_norm.append(tmp)
        new_norm_data_points.append(i)

    new_norm = np.array(new_norm)
    new_norm_data_points = np.array(new_norm_data_points)
    sz = new_norm.shape[0]
    tck, u = scipy.interpolate.splprep([new_norm_data_points, new_norm], s=0)
    xnew, ynew = scipy.interpolate.splev(np.linspace(0, 1, len(rand_lc)), tck, der=0)

    _s0 = []
    _p0 = []

    for i in range(1, len(ynew)-1):
        if (ynew[i]>ynew[i-1]) and (ynew[i]>ynew[i+1]):
            _p0.append(i)
        elif (ynew[i]<ynew[i-1]) and (ynew[i]<ynew[i+1]):
            _s0.append(i)

    max_before_min = 0
    if (xnew[_s0[0]] < xnew[_p0[0]]):
        max_before_min = 1

    _p1 = []

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
    
    _s1 = []

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

    _s2 = []
    _p2 = []

    for i in range(len(_p1)):
        for j in range(len(_s1)-1):
            if (xnew[_s1[j+1]] > xnew[_p1[i]]):
                _s2.append(_s1[j])
                _p2.append(_p1[i])
                break

    _s3 = []
    _p3 = []

    s = np.std(np.array([ynew[_p2[i]] for i in range(len(_p2))]))
    threshold_height = s
    threshold_time = 0                     #! EXTRA PARAMETER SETTED

    for i in range(len(_p2)):
        # x = pythagorean(xnew[_s2[i]], xnew[_p2[i]], ynew[_s2[i]], ynew[_p2[i]])
        y__ = ynew[_p2[i]]-ynew[_s2[i]]
        if (y__ > threshold_height):
            if (xnew[_p2[i]]-xnew[_s2[i]] > threshold_time) or (y__ > s):
                _s3.append(_s2[i])
                _p3.append(_p2[i])

    _s4 = []
    _p4 = []

    s = set()

    for i in range(len(_s3)):
        if _s3[i] in s:
            continue
        s.add(_s3[i])
        _s4.append(_s3[i])
        _p4.append(_p3[i])

    _e0 = []

    for i in range(len(_p4)-1):
        for j in range(len(xnew)):
            if (xnew[j] > xnew[_p4[i]]):
                if (ynew[j] < (ynew[_p4[i]] + ynew[_s4[i]])/2):
                    _e0.append(j)
                    break
                if (xnew[j] > xnew[_s4[i+1]]):
                    _e0.append(j-1)
                    break

    _e0.append(_s2[-1])
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
    st = _zip['start_time']
    pt = _zip['peak_time']
    et =  _zip['end_time']
    pi = _zip['peak_intensity']

    y_min = np.min(ynew)

    final_st = []
    final_pt = []
    final_et = []
    est_et = []
    final_pi = []
    final_err = []

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
        est_et.append(_calc_et + pt[i])

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

    final_zip = pd.DataFrame(zip(final_st, final_pt, final_et, est_et, final_pi, final_err))
    final_zip.columns = ['start_time', 'peak_time', 'end_time', 'est_end_time', 'peak_intensity', 'error']
    # final_zip.sort_values(by=['error'], ascending=True, inplace=True)
    final_zip.to_csv('results_{}.csv'.format(file))

if __name__ == '__main__':
    main()