import numpy as np
from .file_handler import read_lightcurve
from .params import (
    MIN_WINDOW_SIZE,
    MAX_WINDOW_SIZE,
    MINIMUM_DATAPOINTS_IN_GTI,
    SIGMOID_SHIFT_WINDOW_SIZE,
)
from .denoise import smoothening_ma
import scipy
from .flare_detect_minmax import *
from .create_df_minmax import *
from .false_positive_detection import *


class Lightcurve:
    """
    Lightcurve class that accepts file path as argument.
    The main function will execute all denoising and filtering and give out flare details and model fitted parameters that can then be used in the Isolation Classifier to detect false positives.
    """

    def __init__(self, path_to_lc) -> None:
        self.path = path_to_lc
        x_arr, y_arr = read_lightcurve(path_to_lc)
        self.windows = [
            MIN_WINDOW_SIZE
            + (MAX_WINDOW_SIZE - MIN_WINDOW_SIZE)
            * int(1 / (1 + np.exp(-1 * (len(x) - SIGMOID_SHIFT_WINDOW_SIZE))))
            for x in x_arr
        ]
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.x_new = []
        self.y_new = []
        self.flare_details = None

    def main(self, check_false_positives=False, picklePath=None):
        for i in range(len(self.x_arr)):
            if len(self.x_arr[i]) >= MINIMUM_DATAPOINTS_IN_GTI:
                _x, _y = smoothening_ma(
                    self.x_arr[i], self.y_arr[i], 2 * self.windows[i], self.windows[i]
                )
                for j in range(len(_x)):
                    self.x_new.append(_x[j])
                    self.y_new.append(_y[j])
        xnew = np.linspace(
            int(self.x_new[0]),
            int(self.x_new[-1] - self.x_new[0]),
            int(self.x_new[-1] - self.x_new[0]),
        )
        f__ = scipy.interpolate.interp1d(
            self.x_new, self.y_new, fill_value="extrapolate", kind="linear"
        )
        ynew = f__(xnew)
        _s0, _p0 = get_lvl_0_sp_(xnew, ynew, should_plot=False)
        _s1, _p1 = get_lvl_1_sp_(xnew, ynew, _s0, _p0, should_plot=False)
        _s2, _p2 = get_lvl_2_sp_(xnew, ynew, _s1, _p1, should_plot=False)
        _s3, _p3 = get_lvl_3_sp_(xnew, ynew, _s2, _p2, 0.3, should_plot=False)
        _s4, _p4 = get_lvl_4_sp_(xnew, ynew, _s3, _p3, should_plot=False)
        _s5, _p5 = get_lvl_5_sp_(xnew, ynew, _s4, _p4, should_plot=False)

        _e0 = get_lvl_0_e_(xnew, ynew, _s5, _p5, _s0, should_plot=False)
        _e1 = get_lvl_1_e_(xnew, ynew, _s0, _p5, _e0, should_plot=False)

        if len(_e1) != 0:
            h1, h2, h3, h4 = get_interm_zip_features_(ynew, _s5, _p5, _e1)
            if len(h1) != 0:
                _zip = get_interm_zip_(h1, h2, h3, h4)
                g1, g2, g3, g4, g5, g6, g7, g8, g9 = get_final_zip_features(
                    xnew, ynew, _zip
                )
                if len(g1) != 0:
                    final_zip = get_final_zip(g1, g2, g3, g4, g5, g6, g7, g8, g9)
                    """
                    Final Zip contains:  "start_time", "peak_time", "end_time", "est_end_time", "start_intensity", "peak_intensity", "background_counts", "error"
                    """
                    self.flare_details = final_zip
                    model_zip = get_model_features(final_zip, self.path)
                    """
                    Model Zip contains: "_id", "time_ratio", "intensity_ratio", "bandwidth_1", "bandwidth_2", "error"
                    """

        if check_false_positives:
            with_false_positives = predict_false_positives(model_zip, picklePath)
            return with_false_positives
        else:
            return model_zip

    def train_classifier(self, model_zip):
        """
        Needs a dataframe with numerous flares to train
        """
        data = train_isolationForest_(model_zip, force_new_pickle=True)
        return data
