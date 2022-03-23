import os
import zipfile
import re
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table


def unzip_files_pradan_(path_to_files="."):
    """
    Unzips all the files downloaded from Pradan and at the path_to_files
    """
    files = os.listdir(path_to_files)  # Path of root containing all zip XSM files
    # flag = True
    done_zip_files = []
    for file in files:
        if file[-3:] == "zip":
            # flag = False
            with zipfile.ZipFile(file, "r") as zipref:
                if not os.path.isdir(file[:-4]):
                    os.mkdir(file[:-4])
                    zipref.extractall(file[:-4])
            done_zip_files.append(file)


def return_paths_pradan_(file_name, path_to_files="."):
    """
    @returns paths to raw and caliberated folders of that particular day file from pradan downloaded readings
    """
    raw_files_path = f"{path_to_files}/{file_name}/xsm/data/{file_name[8:12]}/{file_name[12:14]}/{file_name[14:16]}/raw"
    calibrated_files_path = f"{path_to_files}/{file_name}/xsm/data/{file_name[8:12]}/{file_name[12:14]}/{file_name[14:16]}/calibrated"
    lc_file_path = f"{path_to_files}/{file_name}/xsm/data/{file_name[8:12]}/{file_name[12:14]}/{file_name[14:16]}/calibrated/{file_name}_level2.lc"
    return lc_file_path, raw_files_path, calibrated_files_path


def txt_to_csv(dfile):
    """Converting the textfile to another textfile where elements are seperated with commas
    Function meant for false detection module only
    """

    with open("resultsdata_mod_.txt", "a") as f:
        f.write("_id,time_ratio,intensity_ratio,bandwidth_1,bandwidth_2,error")
        f.write("\n")

    with open(dfile, "r") as file:
        with open("resultsdata_mod_.txt", "a") as file2:
            rows = file.readlines()
            for row in rows:
                row_commas = re.sub("\s+", ",", row.strip())
                rows_without_ws = row_commas.lstrip()
                file2.write(rows_without_ws)
                file2.write("\n")

    dataframe = pd.read_csv("resultsdata_mod_.txt")
    os.remove("./resultsdata_mod_.txt")
    dataframe.to_csv("resultsdata.csv")
    return dataframe


def read_lightcurve(file, should_plot=False):
    """
    Read LC, ASCII or XLS file
    """
    if file.endswith(".lc"):
        t = Table.read(file)
    elif file.endswith(".xls") or file.endswith(".xlss"):
        xls = pd.ExcelFile(file)
        t = xls.parse(0)
    elif file.endswith(".csv") or file.endswith(".txt"):
        t = pd.read_csv(file)

    tmp_rate = t["RATE"]
    tmp_time = []
    for i in range(len(tmp_rate)):
        tmp_time.append(int(t["TIME"][i] - t["TIME"][0]))

    tmp = 0
    prev = 0
    x_arr = []
    y_arr = []
    __x = []
    __y = []

    plt.figure(figsize=(30, 10))

    for i in range(len(tmp_time) - 1):
        # if discontinuity is greater than 2 timestamps, break into a new interval
        if tmp_time[i + 1] - tmp_time[i] > 2:
            __x = []
            __y = []
            for j in range(prev, i + 1):
                __x.append(int(tmp_time[j]))
                __y.append(int(tmp_rate[j]))
                # if discontinuity is of 2, make the mid as average of the ends
                if tmp_time[j + 1] - tmp_time[j] == 2:
                    __x.append(int(tmp_time[j] + 1))
                    __y.append(int((tmp_rate[j] + tmp_rate[j + 1]) / 2))
            x_arr.append(__x)
            y_arr.append(__y)
            prev = i + 1
            if should_plot:
                plt.scatter(__x, __y, s=0.3)
    # because we are missing out the last interval, doing it seperately
    __x = []
    __y = []
    for i in range(prev, len(tmp_time) - 1):
        __x.append(int(tmp_time[i]))
        __y.append(int(tmp_rate[i]))
    x_arr.append(__x)
    y_arr.append(__y)
    if should_plot:
        plt.scatter(__x, __y, s=0.3)
        plt.grid()
        plt.show()

    for i in x_arr:
        tmp += len(i)

    return x_arr, y_arr
