import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compress_time_stamps(lc_example, window_size=60, should_plot=False, s=2):
    """
    Helper function for moving window smoothening functions
    """
    new_lc = np.ones(int(len(lc_example) / window_size) + 1)
    for i in range(0, len(lc_example), window_size):
        new_lc[int(i / window_size)] *= lc_example[i]
    if should_plot:
        plt.figure(figsize=(25, 7))
        plt.grid()
        plt.scatter(range(len(new_lc)), new_lc, s=s)
    return new_lc


def start_thresh_algo(lc, window_size=4, slope_threshold=20, num_frames_threshold=4):
    """
    Detects the start time of flares
    """
    counter = 0
    flare_start_time = []
    allowed_discrepancies = 0.2
    for i in range(0, len(lc)):
        if i + window_size >= len(lc):
            index = len(lc) - 1
        else:
            index = i + window_size
        slope = (lc[index] - lc[i]) / window_size
        if slope >= slope_threshold:
            counter += 1
            discrepancies = 0
            for j in range(i + 1, i + window_size):
                if j >= len(lc):
                    continue
                # I'm assuming that moving average did the smoothing, i.e. now every flare rise will only have a rise, with a bit discrepancy
                if lc[j] < lc[j - 1]:
                    discrepancies += 1
            if (
                discrepancies < allowed_discrepancies * window_size
                and counter >= num_frames_threshold
            ):
                if (
                    lc[num_frames_threshold * window_size + i] - lc[i]
                    > slope * num_frames_threshold
                ):
                    flare_start_time.append(i)
                i += window_size * num_frames_threshold
                counter = 0
    return np.array(flare_start_time)


def start_thresh_algo_helper(lc, startTimes):
    for i, n in enumerate(startTimes):
        if i + 1 >= len(startTimes):
            index = len(startTimes) - 1
        else:
            index = i + 1
        if startTimes[index] < startTimes[i] + 9:
            startTimes[index] = startTimes[i]
    onesAtStartTime = np.zeros(len(lc))
    onesAtStartTime[startTimes] = int(np.mean(lc) * 2 / 3)

    return onesAtStartTime


def end_thresh_algo(lc_compressed, startTimes):
    endTimes = []
    for i, startTime in enumerate(startTimes):
        y_val = lc_compressed[startTime]
        counter = startTime + 1
        tmp = startTimes[i + 1] if i + 1 < len(startTimes) else len(lc_compressed)
        peakReached = False
        while (
            counter < len(lc_compressed) - 1
            and lc_compressed[counter] > y_val
            and counter < tmp
        ):
            if counter > len(lc_compressed):
                counter = len(lc_compressed) - 1
            counter += 1
        endTimes.append(counter)
    return endTimes


def end_thresh_algo_helper(lc_compressed, endTimes):
    onesAtEndTimes = np.zeros(len(lc_compressed))
    onesAtEndTimes[endTimes] = int(np.mean(lc_compressed))
    return onesAtEndTimes


def cleaned_dataframe(startTimes, endTimes, window_median):
    df = pd.DataFrame(zip(startTimes, endTimes))
    df.columns = ["Start Time", "End Time"]
    df["Start Time"] *= window_median
    df["End Time"] *= window_median
    drops = []
    # dropping out repeated start times
    for i, startTime in enumerate(np.array(df["Start Time"])):
        if i + 1 < len(df["Start Time"]):
            if df["Start Time"].iloc[i + 1] == startTime:
                drops.append(i)
    df.drop(drops, axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def final_cleaned_durations(df, window_median):
    Times = df.to_numpy(dtype=int)
    startTimes_ = Times[:, 0] // window_median
    endTimes_ = Times[:, 1] // window_median
    return startTimes_, endTimes_
