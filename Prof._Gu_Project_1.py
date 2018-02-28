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

#####################
###hyperparameters###
#####################

data_points_per_series=5
degree_for_fitting=3


raw_data=pd.read_csv("C:/Users/aminv/Desktop/GSPC.csv")

#date from old to new
daily_price_data = raw_data[['Date','Close']]


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

def plot_fits(data):
    for index, row in data.iterrows():
        coefficients=row[1:]
        curve = poly.Polynomial(coefficients[::-1])
        x = np.linspace(0, 5, 256, endpoint=True)
        curves=plt.figure("curves")
        curves = plt.plot(x, curve(x))
    return curves

def plot_4d(data):
    fig = plt.figure("4d_parameters")
    ax = fig.add_subplot(111, projection='3d')
    
    x = data['0']
    y = data['1']
    z = data['2']
    c = data['3']

    ax.scatter(x, y, z, c=c, cmap=plt.hot())
    plt.show()
        
        
        
price_series_data=build_series(daily_price_data, data_points_per_series)
fit_parameters=fit(price_series_data, degree_for_fitting)
plot_fits(fit_parameters)
if(degree_for_fitting==3):
    plot_4d(fit_parameters)


