# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:52:39 2018

@author: aminv
"""

import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

import scipy.linalg

#####################
###hyperparameters###
#####################

data_points_per_series=5
degree_for_fitting=3
n_clusters=3



raw_data=pd.read_csv("C:/Users/aminv/Desktop/S&P500Data5Y.csv")

#date from old to new
daily_price_data = raw_data[['Date','Close']]

#for plotting
colors=['k','c','m','r','g','b']


def build_series(data, data_points_per_series):
    columns=range(data_points_per_series)
    columns=[str(i) for i in columns]
    columns = ['StartDate']+columns
    price_series_data=pd.DataFrame(columns=columns)
    for i in range((data.shape[0])/data_points_per_series):
        row=[daily_price_data['Date'][i*data_points_per_series]]
        series=[daily_price_data['Close'][i*data_points_per_series:(i*data_points_per_series)+data_points_per_series]]
        series=series[0].tolist()
        row=row+series
        price_series_data.loc[price_series_data.shape[0]] = row
    return price_series_data
        
def fit(data, degree_for_fitting):
    columns=range(degree_for_fitting+1)
    columns=[str(i) for i in columns]
    columns = ['StartDate']+columns
    fit_parameters=pd.DataFrame(columns=columns)
    for i in range(data.shape[0]):
        x=range(data_points_per_series)
        y=data.iloc[i][1:]
        coefficients=np.polyfit(x, y, degree_for_fitting, rcond=None, full=False, w=None, cov=False)
        row = [data['StartDate'][i]] + coefficients.tolist()
        fit_parameters.loc[fit_parameters.shape[0]] = row
    return fit_parameters

def plot_fits(data, color='k'):
    for index, row in data.iterrows():
        coefficients=row[1:]
        curve = poly.Polynomial(coefficients[::-1])
        x = np.linspace(0, 5, 256, endpoint=True)
        curves=plt.figure("curves")
        curves = plt.plot(x, curve(x), color=color)
    return curves

def fit_plane(df):
    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(df['0'])):
        tmp_A.append([df['0'][i], df['1'][i], 1])
        tmp_b.append(df['2'][i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)
    
    print "solution:"
    print "%f x + %f y + %f = z" % (fit[0], fit[1], fit[2])
    print "errors:"
    print errors
    print "residual:"
    print residual
    return fit


def plot_4d(data, plane=False):
    fig = plt.figure("4d_parameters")
    ax = fig.add_subplot(111, projection='3d')
    
    x = data['0']
    y = data['1']
    z = data['2']
    c = data['3']

    ax.scatter(x, y, z, c=c, cmap=plt.hot())
    
    if(plane):
        # plot plane
        fit=fit_plane(data)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                          np.arange(ylim[0], ylim[1]))
        Z = np.zeros(X.shape)
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
        ax.plot_wireframe(X,Y,Z, color='k')
    
    plt.show()



    
        
        
price_series_data=build_series(daily_price_data, data_points_per_series)
fit_parameters=fit(price_series_data, degree_for_fitting)
plot_fits(fit_parameters)
if(degree_for_fitting==3):
    plot_4d(fit_parameters, plane=True)
    
    
def k_means_clustering(df):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df)
    
    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_
    
    df['Labels'] = labels
    
    if(degree_for_fitting==3):
        fig = plt.figure("k_means_clustering")
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['0'], df['1'], df['2'], c=df['Labels'], cmap=plt.cm.tab10, alpha=0.5, edgecolor='k')
        for centroid in centroids:
            ax.scatter(centroid[0],centroid[1],centroid[2],c='black')
        plt.show()
    return df

data_for_clustering=fit_parameters.loc[:, fit_parameters.columns != 'StartDate']
data_for_clustering=data_for_clustering.loc[:, data_for_clustering.columns != '3']

df=k_means_clustering(data_for_clustering)

Cluster_curves=plt.figure("Cluster_curves")
clusters=[]
for i in range(n_clusters):
    clusters=clusters+[df.loc[df['Labels'] == i]]
    for index, row in clusters[i].iterrows():
        coefficients=row[:-1]
        curve = poly.Polynomial(coefficients[::-1])
        x = np.linspace(0, 5, 256, endpoint=True)
        Cluster_curves = plt.plot(x, curve(x), color=colors[i])
plt.show()
    
