# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:52:39 2018

@author: aminv
"""

import os
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

import scipy.linalg




def build_series(data, data_points_per_series):
    """
        Subdivides a given dataframe of data into a bunch of series with data_points_per_series data points per series.
    """
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
    """
        fits an degree_for_fitting polynomial to a given series, data.
    """
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

def plot_curves(data, color='k'):
    """
        input is a dataframe where each row is the coefficients of a polynomial in order of highest to lowest degree.
        loops through the dataframe and plots each curve.
        default color is black can be changed by setting color
    """
    for index, row in data.iterrows():
        coefficients=row
        curve = poly.Polynomial(coefficients[::-1])
        x = np.linspace(0, 5, 256, endpoint=True)
        curves=plt.figure("curves")
        curves = plt.plot(x, curve(x), color=color)
    return curves

def fit_plane(df):
    """
        fits a plane to 3 dimensional data. Input is a pandas dataframe.
        1st column is first variable...
    """
    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(df.iloc[:,0])):
        tmp_A.append([df.iloc[:,0][i], df.iloc[:,1][i], 1])
        tmp_b.append(df.iloc[:,2][i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)
    
    print filepath[filepath.rfind("/")+1:filepath.rfind(".csv")]
    print "solution:"
    print "%f x + %f y + %f = z" % (fit[0], fit[1], fit[2])
    return fit


def plot_4d(data, plane=False):
    """
        plots four dimensional data of a given dataframe, data, in three dimensional space 
        and a color space. 
        
        1st column is the x axis, 2nd y axis, 3rd z axis, and the fourth is the color.
        
        if plane = true fits a plane to the data and plots it
    """
    fig = plt.figure("4d_parameters")
    ax = fig.add_subplot(111, projection='3d')
    
    x = data.iloc[:,0]
    y = data.iloc[:,1]
    z = data.iloc[:,2]
    c = data.iloc[:,3]

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

def pca(X=np.array([]), no_dims=2):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

    

def k_means_clustering(df):
    """
        Runs K means clustering on a given pandas dataframe 
        where each row is a data point and each column is a variable.
        
        Built in plotting if degree_for_fitting = 3 uses first 3 column as x, y, z axis respectivly
        and colors to distinguish the clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df)
    
    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_
    
    df['Labels'] = labels
    
    if(degree_for_fitting==3):
        fig = plt.figure("k_means_clustering")
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], c=df['Labels'], cmap=plt.cm.tab10, alpha=0.5, edgecolor='k')
        for centroid in centroids:
            ax.scatter(centroid[0],centroid[1],centroid[2],c='black')
        plt.show()
    return df

def main(filepath, data_points_per_series=5, degree_for_fitting=3, n_clusters=3):

    raw_data=pd.read_csv(filepath)
    
    #date from old to new
    daily_price_data = raw_data[['Date','Close']]
    
    #for plotting
    colors=['k','c','m','r','g','b']      
            
    price_series_data=build_series(daily_price_data, data_points_per_series)
    fit_parameters=fit(price_series_data, degree_for_fitting)
    fit_parameters_without_date=fit_parameters.loc[:, fit_parameters.columns != 'StartDate']
    plot_curves(fit_parameters_without_date)
    if(degree_for_fitting==3):
        plot_4d(fit_parameters_without_date, plane=True)
        
        
    
    
    data_for_clustering=fit_parameters_without_date
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

######################
####Function Calls####
######################

rootdir="C:/Users/aminv/Desktop/StockData"
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + "/" + file

        if filepath.endswith(".csv"):
            print (filepath)
            main(filepath)
            
            
            
            
            
            
            
            