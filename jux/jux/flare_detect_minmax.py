import numpy as np
import matplotlib.pyplot as plt
from .helper import get_slope
from .params import MIN_MULTIPLIER, THRESHOLD_Y_NOISE_LVL5, THRESHOLD_X_NOISE_LVL5

"""Getting extremas functions"""


def get_lvl_0_sp_(xnew, ynew, should_plot=False):
    """
    Detection of all Minimas and Maximas
    """

    _s0 = []
    _p0 = []
    if ynew[0] <= ynew[1]:
        _s0.append(0)
    for i in range(1, len(ynew) - 1):
        if (ynew[i] > ynew[i - 1]) and (ynew[i] > ynew[i + 1]):
            _p0.append(i)
        elif (ynew[i] < ynew[i - 1]) and (ynew[i] < ynew[i + 1]):
            _s0.append(i)
    if ynew[-2] >= ynew[-1]:
        _s0.append(len(xnew) - 1)
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Level 0 Maximas")
        plt.plot([xnew[i] for i in _p0], [ynew[i] for i in _p0], "o", xnew, ynew)
        plt.show()
        plt.figure(figsize=(30, 10))
        plt.title("Level 0 Minimas")
        plt.plot([xnew[i] for i in _s0], [ynew[i] for i in _s0], "o", xnew, ynew)
        plt.show()
    return _s0, _p0


def get_lvl_1_sp_(xnew, ynew, _s0, _p0, should_plot=False):
    _s1 = []
    _p1 = []
    for i in range(len(_p0)):
        for j in range(len(_s0) - 1):
            # pairing of minimas and maximas as starts and peaks
            if xnew[_s0[j + 1]] > xnew[_p0[i]]:
                _s1.append(_s0[j])
                _p1.append(_p0[i])
                break
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Level 1 Maximas")
        plt.plot([xnew[i] for i in _p1], [ynew[i] for i in _p1], "o", xnew, ynew)
        plt.show()
        plt.figure(figsize=(30, 10))
        plt.title("Level 1 Minimas")
        plt.plot([xnew[i] for i in _s1], [ynew[i] for i in _s1], "o", xnew, ynew)
        plt.show()
        print(len(_s1))
    return _s1, _p1


def get_lvl_2_sp_(xnew, ynew, _s1, _p1, should_plot=False):
    _s2 = []
    _p2 = []
    _slopes = np.array(
        [
            get_slope(xnew[_s1[i]], xnew[_p1[i]], ynew[_s1[i]], ynew[_p1[i]])
            for i in range(len(_s1))
        ]
    )
    mean_sl = np.mean(_slopes)
    for i in range(len(_s1)):
        # slope thresholding, the "significant" flares rise slopes are larger than mean rise
        if _slopes[i] > mean_sl:
            _s2.append(_s1[i])
            _p2.append(_p1[i])
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Level 2 Maximas")
        plt.plot([xnew[i] for i in _p2], [ynew[i] for i in _p2], "o", xnew, ynew)
        plt.show()
        plt.figure(figsize=(30, 10))
        plt.title("Level 2 Minimas")
        plt.plot([xnew[i] for i in _s2], [ynew[i] for i in _s2], "o", xnew, ynew)
        plt.show()
        print(len(_s2))
    return _s2, _p2


def get_lvl_3_sp_(xnew, ynew, _s2, _p2, f, should_plot=False):
    _s3 = []
    _p3 = []
    # setting a height threshold
    _std = np.std(np.array([ynew[_p2[i]] - ynew[_s2[i]] for i in range(len(_s2))]))
    for i in range(len(_s2)):
        if ynew[_p2[i]] - ynew[_s2[i]] > _std * f:  #! Hyperparameter
            _s3.append(_s2[i])
            _p3.append(_p2[i])
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Level 3 Maximas")
        plt.plot([xnew[i] for i in _p3], [ynew[i] for i in _p3], "o", xnew, ynew)
        plt.show()
        plt.figure(figsize=(30, 10))
        plt.title("Level 3 Minimas")
        plt.plot([xnew[i] for i in _s3], [ynew[i] for i in _s3], "o", xnew, ynew)
        plt.show()
    return _s3, _p3


def get_lvl_4_sp_(xnew, ynew, _s3, _p3, should_plot=False):
    _s4 = []
    _p4 = []
    s = set()
    for i in range(len(_s3)):
        # appending and pairing unique starts and peaks
        if _s3[i] in s:
            continue
        s.add(_s3[i])
        _s4.append(_s3[i])
        _p4.append(_p3[i])
    if should_plot:
        plt.figure()
        plt.title("Final Start and Peak")
        plt.plot(
            [xnew[i] for i in _p4],
            [ynew[i] for i in _p4],
            "o",
            [xnew[i] for i in _s4],
            [ynew[i] for i in _s4],
            "x",
            xnew,
            ynew,
        )
        plt.show()
    return _s4, _p4


def get_lvl_5_sp_(xnew, ynew, _s4, _p4, should_plot=False):
    # a filter for too close peaks
    li = []
    for i in range(len(_p4) - 3):
        if (xnew[_p4[i + 1]] - xnew[_p4[i]] < THRESHOLD_X_NOISE_LVL5) and (
            np.abs(ynew[_p4[i + 1]] - ynew[_p4[i]]) < THRESHOLD_Y_NOISE_LVL5
        ):
            if ynew[_p4[i + 1]] > ynew[_p4[i]]:
                li.append(i)
            elif ynew[_p4[i + 1]] < ynew[_p4[i]]:
                li.append(i+1)
    for index in sorted(li, reverse=True):
        del _s4[index]
        del _p4[index]
    if should_plot:
        plt.figure(figsize=(30, 10))
        plt.title("Final Start and Peak")
        plt.plot(
            [xnew[i] for i in _p4],
            [ynew[i] for i in _p4],
            "o",
            [xnew[i] for i in _s4],
            [ynew[i] for i in _s4],
            "x",
            xnew,
            ynew,
        )
        plt.show()
    return _s4, _p4


def get_lvl_0_e_(xnew, ynew, _s5, _p5, _s0, should_plot=False):
    _e0 = []
    for i in range(len(_p5)):
        for j in range(_p5[i], len(xnew)):
            # edge case for the last peak's end
            if i == len(_p5) - 1:
                if xnew[_s0[-1]] < xnew[_p5[i]]:
                    _e0.append(len(xnew) - 1)
                    break
                elif j == len(xnew) - 1:
                    _e0.append(len(xnew) - 1)
            # usual ends with definition of being midway the intensity of peak and start
            if ynew[j] < (ynew[_p5[i]] + ynew[_s5[i]]) / 2:
                _e0.append(j)
                break
            if i + 1 < len(_s5):
                if xnew[j] > xnew[_s5[i + 1]]:
                    _e0.append(j - 1)
                    break
    if should_plot:
        plt.figure()
        plt.title("Level 0 Ends")
        plt.plot(
            [xnew[i] for i in _p5],
            [ynew[i] for i in _p5],
            "o",
            [xnew[i] for i in _e0],
            [ynew[i] for i in _e0],
            "x",
            xnew,
            ynew,
        )
        plt.show()
    return _e0


def get_lvl_1_e_(xnew, ynew, _s0, _p5, _e0, should_plot=False):
    _e1 = []
    for i in range(len(_e0)):
        # for each _e0
        if ynew[_e0[i]] < ynew[_p5[i]]:
            _e1.append(_e0[i])
        else:
            # if intensity of end > its peak
            # find the next minima that has it's reverse true
            for j in range(len(_s0)):
                if xnew[_s0[j]] > xnew[_p5[i]]:
                    if ynew[_s0[j + 1]] > ynew[_s0[j]]:
                        _e1.append(_s0[j])
                        break
    if should_plot:
        plt.figure()
        plt.title("Peaks and Ends")
        plt.plot(
            [xnew[i] for i in _p5],
            [ynew[i] for i in _p5],
            "o",
            [xnew[i] for i in _e1],
            [ynew[i] for i in _e1],
            "x",
            xnew,
            ynew,
        )
        plt.show()
    _e1 = np.array(_e1)
    return _e1
