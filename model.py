import re
# from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

with open('resultsdata_mod.txt','a') as f:
    f.write('time_ratio,intensity_ratio,bandwidth,error')
    f.write('\n')

"""Converting the textfile to another textfile where elements are seperated with commas"""
with open('resultsdata.txt', 'r') as file:
    with open('resultsdata_mod.txt', 'a') as file2:
        rows = file.readlines()
        for row in rows:
            row_commas = re.sub('\s+', ',', row.strip())
            file2.write(row_commas)
            file2.write('\n')

dataframe = pd.read_csv('resultsdata_mod.txt')
dataframe.to_csv('resultsdata_mod.csv')

"""Reading the Pandas dataframe"""
data = pd.read_csv('resultsdata_mod.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)


"""Dimensionality Reduction using PCA -> 2 features namely PCA_1 and PCA_2"""
pca = PCA(n_components=2, svd_solver='randomized')
X_pca = pca.fit_transform(data)


data_pca = pd.DataFrame(X_pca)
data_pca.columns = ['PCA_1', 'PCA_2']


"""Initializing the Classifier"""
isf = IsolationForest(n_estimators=100, contamination=0.025)
# params = {'' : [0.2, 0.02, 0.001, 0.1],
#           'n_estimators': [100, 200]}
# clf = GridSearchCV(isf2, param_grid=params, scoring='accuracy', n_jobs=-1)
# clf.fit(data)
# print(clf.best_params_)
preds = isf.fit_predict(data_pca[['PCA_1', 'PCA_2']])
data_pca['outliers'] = preds.astype(str)
data_pca['outlier_score'] = isf.decision_function(data_pca[['PCA_1','PCA_2']])
data_pca['outliers'] = data_pca['outliers'].map({'1': 'Normal_Flares', '-1': 'Outliers'})
print(data_pca['outliers'].value_counts())


def scatter_plot(dataframe, x, y, color, title, hover_name):
    fig = px.scatter(dataframe, x = x, y=y,
                    color = color,
                     hover_name = hover_name)
    #set the layout conditions
    fig.update_layout(title = title,
                     title_x = 0.5)
    #show the figure
    fig.show()
#create scatter plot
scatter_plot(data_pca, "PCA_1", "PCA_2", "outliers",
             "Solar Flares Outlier Detection", hover_name=None)