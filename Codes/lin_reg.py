import numpy as np
import matplotlib.pyplot as plt
import itertools
import utils as us
import importlib
from scipy.stats import norm

importlib.reload(us)

weight_lin_reg = [3, 5, -7, -2, 7]
b_lin_reg = 0
# ranges_lin_reg = [4, 5, 8]
ranges_lin_reg = [1,5,2, 3,2]
# ranges_lin_reg = [2,2,2, 2,2]
weight_lin_reg = [3, 5, -7, -2, 6]

# b_vec_lin_reg = [0, 2, 8,5,1]
b_vec_lin_reg = [0, 0, 0,0,0]


def feature(position):
    return " {" + str(position) + ":3d} |"
def y_out(position):
    return "{" + str(position) + ":4.2f} |"
def vimp(position):
    return " {" + str(position) + ":3.3f} |"

# feature = " {" + str(position) + ":3d} |"
# y = "{" + str(position) + ":4.2f} |"
# vimp = " {" + str(position) + ":3.3f} |"
# for i in range(0, size):
    #     print("local feature {} importance: {}".format(i+1, round(abs(abs(val[i] * weight[i])/tmp), 3)))
    # str = ""
    # position=0
    #
    # for i in range(0,size):
    #     str += feature(position)
    #     position += 1
    # str += y_out(position)
    # position += 1
    # for i in range(0,size):
    #     str+=vimp(position)
    # print("{0:3d} | {1:3d} | {2:3d} | {3:4.2f} | {4:3.3f} | {5:3.3f} | {6:3.3f} |".format(val[0], val[1], val[2], tmp,abs(abs(val[0] * weight[0])/tmp),abs(abs(val[1] * weight[1])/tmp),abs(abs(val[2] * weight[2])/tmp) ))


def linear_function(val = [5,2]):

    global weight_lin_reg
    global b_lin_reg
    weight = weight_lin_reg
    b = b_lin_reg

    tmp = 0
    abs_tmp = 0
    size = len(val)
    for i in range(0,size):
        tmp += val[i] * weight[i]
        abs_tmp += abs(val[i] * weight[i])

    tmp += b
    abs_tmp += b
    return tmp, abs_tmp

def vimp_lf(sample, abs_sum):

    global weight_lin_reg
    weight = weight_lin_reg

    nfeatures = len(sample)
    vimp = np.zeros(nfeatures)
    for i in range(0, nfeatures):
        vimp[i] = abs(weight[i] * sample[i]) / abs_sum
        # vimp[i] = abs(weight[i]) / abs_sum

    return vimp

def lf_dataset(nsamples = 150, with_vimp = True):
    np.random.seed(0)

    global weight_lin_reg
    global b_lin_reg
    global ranges_lin_reg
    global b_vec_lin_reg
    weight = weight_lin_reg
    b = b_lin_reg
    ranges = ranges_lin_reg
    b_vec = b_vec_lin_reg

    # weight = [3, 5, -7]
    # ranges = [4,5,8]
    # b_vec = [1,0,0]
    # b = 0

    nfeatures = len(weight)
    X = np.ndarray(shape=(nsamples, nfeatures), dtype=float)

    for i in range(0, nfeatures):
        X[:,i] = (np.random.uniform(0, 1, nsamples) * ranges[i]) + b_vec[i]

    t = 'R'

    value = np.zeros(shape=(nsamples, 1), dtype=float)
    abs_sum = np.zeros(shape=(nsamples, 1), dtype=float)
    for i in range(0, nsamples):
        value[i], abs_sum[i] = linear_function(X[i, :])

    vimp = np.zeros((nsamples, nfeatures))
    for i in range(0,nsamples):
        vimp[i,:] = vimp_lf(X[i,:], abs_sum[i])

    if with_vimp:
        return X,value,t, vimp
    else:
        return X,value,t


def vimp_lr(X):

    global weight_lin_reg
    global b_lin_reg
    weight = weight_lin_reg
    b = b_lin_reg

    nsamples = X.shape[0]
    nfeatures = X.shape[1]
    value = np.zeros(shape=(nsamples, 1), dtype=float)
    abs_sum = np.zeros(shape=(nsamples, 1), dtype=float)
    for i in range(0, nsamples):
        value[i], abs_sum[i] = linear_function(X[i, :])

    vimp = np.zeros((nsamples, nfeatures))
    for i in range(0, nsamples):
        vimp[i, :] = vimp_lf(X[i, :], abs_sum[i])
    return vimp