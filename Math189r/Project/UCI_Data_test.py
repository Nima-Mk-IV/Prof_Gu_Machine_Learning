# -*- coding: utf-8 -*-
"""
Created on Thu May 17 19:38:40 2018

@author: aminv
"""

import os
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import pylab
import scipy.linalg
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn import svm
from sklearn.metrics import accuracy_score

import pickle

###################################
##########T-sne and PCA############
###################################

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Processing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.                                    # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


###################################
########Plotting functions#########
###################################
wdir='C:/Users/aminv/Desktop/UCI_Smartphone_Dataset'

X_train=pd.read_csv(wdir+'/Train/X_train.txt',' ')
y_train=pd.read_csv(wdir+'/Train/y_train.txt')
subject_id_train=pd.read_csv(wdir+'/Train/subject_id_train.txt')

def fit(data, degree_for_fitting):
    """
        fits an degree_for_fitting polynomial to a given series, data.
    """
    columns=range(degree_for_fitting+1)
    columns=[str(i) for i in columns]
    
    fit_parameters=pd.DataFrame(columns=columns)
    for i in range(data.shape[0]):
        x=range(data.shape[1])
        y=data.iloc[i]
        coefficients=np.polyfit(x, y, degree_for_fitting, rcond=None, full=False, w=None, cov=False)
        row = coefficients.tolist()
        fit_parameters.loc[fit_parameters.shape[0]] = row
    return fit_parameters

#fits=fit(X_train, 10)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_color_curves(data, lables):
    """
        input is a dataframe where each row is the coefficients of a polynomial in order of highest to lowest degree.
        loops through the dataframe and plots each curve.
        default color is black can be changed by setting color
    """
    cmap=get_cmap(lables.nunique())
    #print('walking='+ str(cmap(1)))
    for index, row in data.iterrows():
        coefficients=row
        curve = poly.Polynomial(coefficients[::-1])
        x = np.linspace(0, 5, 256, endpoint=True)
        curves=plt.figure("curves")
        curves = plt.plot(x, curve(x), color=cmap(lables.iloc[index,0]-1))
    rect1 = matplotlib.patches.Rectangle((00,0), 100, 100, color=cmap(1))
    curves.add_patch(rect1)
    return curves

#plot_color_curves(X_train, y_train)
    
#X_train_tsne=tsne(X_train.values, initial_dims=561)

def plot_2d(X,y,figure):
    ''' Gives a 2D scatter plot of X 
    and colors it using a color map with the lables of y'''
    cmap=get_cmap(y_train.nunique())
    y_colors=cmap(y_train.values)
    y_colors=y_colors[:,0,:]
    plt.figure(figure)
    plt.scatter(X[:,0],X[:,1],c=y_colors)
    
    classes=['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']
    class_colors=cmap(range(1,13))
    recs = []
    for i in range(0,len(class_colors)):
           recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colors[i]))
    plt.legend(recs,classes,loc=4)

'''    
with open('Tsne_X_train.pickle', 'wb') as handle:
            pickle.dump(X_train_tsne, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''    
            
def opt_SVM(X,y):
    X=X.values
    y=y.values
    data_rows=X.shape[0]
    y_train=y[:int(data_rows*(2.0/3.0))]
    y_test=y[int(data_rows*(2.0/3.0)):]
    
    acc_list=[]
    for i in range(2,X.shape[1],10):
        X_svm=pca(X, no_dims=i)
        X_train=X_svm[:int(data_rows*(2.0/3.0)),:]
        X_test=X_svm[int(data_rows*(2.0/3.0)):,:]
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(X_train, y_train)
        clf.decision_function_shape = "ovr"
        y_pred = clf.predict(X_test)
        acc_list=acc_list+[accuracy_score(y_test, y_pred)]
        print('Dimension for SVM: ' + str(i) +', gives accuracy: ' + str(acc_list[-1]))
    
    return np.argmax(acc_list)*10 + 2, acc_list
            
#X_train_pca=pca(X_train,no_dims=2)


###################################
##########Hyperparameters##########
###################################
method=''
final_dims=2
svm_dim=50
###################################
########       Main       #########
###################################
if __name__ == "__main__":
    print('Running: '+method+'...')
    if 'PCA' in method:
        X_train_pca=pca(X_train,no_dims=final_dims)
        if final_dims==2:
            print('Plotting Proccessed Data...')
            plot_2d(X_train_pca, y_train, 'pca')
            print('Done')
            
    if 'tSNE' in method:
        X_train_tsne=tsne(X_train.values, no_dims=final_dims, initial_dims=561,)
        if final_dims==2:
            print('Plotting Proccessed Data...')
            plot_2d(X_train_pca, y_train, 'tsne')
            print('Done')
            
    if 'SVM' in method:
        print('Running PCA...')
        X_svm=pca(X_train, no_dims=svm_dim)
        X_train_svm=X_svm[:5000,:]
        X_test_svm=X_svm[5000:,:]
        y_train_svm=y_train[:5000]
        y_test_svm=y_train[5000:]
        print('Training SVM classifier...')
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(X_train_svm, y_train_svm)
        clf.decision_function_shape = "ovr"
        print('Predicting SVM classifier...')
        y_pred_svm = clf.predict(X_test_svm)
        print (accuracy_score(y_test_svm, y_pred_svm) )
		
		


'''		
fig = plt.figure()
plt.plot(range(2,X_train.shape[1],10),acc_list)
plt.xlabel('Number of Dimensions')
plt.ylabel('Accuracy')
'''