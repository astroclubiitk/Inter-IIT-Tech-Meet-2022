import re
import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from .file_handler import txt_to_csv
import pickle
import datetime
from .params import IF_N_ESTIMATORS, IF_CONTAMINATION, IF_MAX_FEATURES


def get_model_features(final_zip, file):
    """
    Model features packaging
    """
    f0 = []
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    f5 = []
    for i in range(len(final_zip)):
        f0.append(
            str((final_zip["start_time"][i] + final_zip["end_time"][i]) // 2)
            + "_"
            + file
        )
        t1 = (final_zip["est_end_time"][i] - final_zip["start_time"][i]) / (
            final_zip["end_time"][i] - final_zip["peak_time"][i]
        )
        f1.append(t1 * t1)
        t2 = (final_zip["peak_intensity"][i] - final_zip["background_counts"][i]) / (
            final_zip["background_counts"][i] - final_zip["start_intensity"][i]
        )
        f2.append(t2)
        t3 = (final_zip["peak_intensity"][i] - final_zip["start_intensity"][i]) / (
            final_zip["peak_time"][i] - final_zip["start_time"][i]
        )
        f3.append(t3)
        t4 = final_zip["error"][i]
        f4.append(t4)
        t5 = (final_zip["peak_intensity"][i] - final_zip["start_intensity"][i]) / (
            final_zip["est_end_time"][i] - final_zip["peak_time"][i]
        )
        f5.append(t5)
    tmp = pd.DataFrame(zip(f0, f1, f2, f3, f5, f4))
    tmp.columns = [
        "_id",
        "time_ratio",
        "intensity_ratio",
        "bandwidth_1",
        "bandwidth_2",
        "error",
    ]
    return tmp


def train_isolationForest_(data, force_new_pickle=False):
    """
    Need the data frame with "Unnamed: 0", "_id", "time_ratio", "intensity_ratio", "bandwidth_1", "bandwidth_2", "error"
    """
    try:
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    except:
        assert "Dataframe wasn't formed by pandas, couldn't remove Unnamed: 0 column"
    data.dropna(axis=0)
    id = data["_id"]
    data.drop(["_id"], axis=1, inplace=True)

    if os.path.exists("isf.pickle") and not force_new_pickle:
        with open("isf.pickle", "rb") as f:
            isf = pickle.load(f)
    else:
        isf = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION,
            random_state=42,
            n_jobs=-1,
            max_features=IF_MAX_FEATURES,
            bootstrap=False,
        )
        isf.fit(
            data[
                ["time_ratio", "intensity_ratio", "bandwidth_1", "bandwidth_2", "error"]
            ]
        )

    preds = isf.predict(
        data[["time_ratio", "intensity_ratio", "bandwidth_1", "bandwidth_2", "error"]]
    )
    data["outliers"] = preds.astype(str)

    data["outlier_score"] = isf.decision_function(
        data[["time_ratio", "intensity_ratio", "bandwidth_1", "bandwidth_2", "error"]]
    )
    data["outliers"] = data["outliers"].map({"1": "Normal_Flares", "-1": "Outliers"})
    data["_id"] = id

    if force_new_pickle:
        with open("isf.pickle", "wb") as f:
            pickle.dump(isf, f)
    return data


def predict_false_positives(data, picklePath):
    """
    Needs a pickle with Isolation Forest Object saved. Also needs a dataframe 'data' with the flare details
    """
    with open(picklePath, "rb") as f:
        isf = pickle.load(f)

    id = data["_id"]
    data.drop(["_id"], axis=1, inplace=True)

    """Dimensionality Reduction using PCA -> 2 features namely PCA_1 and PCA_2"""
    pca = PCA(n_components=2, svd_solver="randomized", whiten=True)
    X_pca = pca.fit_transform(data)
    data_pca = pd.DataFrame(X_pca)
    data_pca.columns = ["PCA_1", "PCA_2"]

    preds = isf.predict(data_pca[["PCA_1", "PCA_2"]])
    data["outliers"] = preds.astype(str)

    data_pca["outlier_score"] = isf.decision_function(data_pca[["PCA_1", "PCA_2"]])
    data["outliers"] = data["outliers"].map({"1": "Normal_Flares", "-1": "Outliers"})
    data["_id"] = id
    return data


def scatter_plot_(dataframe, x, y, color, title, hover_name):
    fig = px.scatter(dataframe, x=x, y=y, color=color, hover_name=hover_name)
    # set the layout conditions
    fig.update_layout(title=title, title_x=0.5)
    # show the figure
    fig.show()
