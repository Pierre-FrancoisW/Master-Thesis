import numpy as np
from sklearn.utils import check_random_state
import shap
import itertools
import lin_reg as lreg
import pandas as pd
import importlib
def pprint(n,vec):
    nv = len(vec)
    print("{:<20}".format(n),end="")
    for i in range(nv):
        print('{:<7.3f}'.format(np.round(vec[i],decimals=3)),end="")
    print('\n',end="")

def make_led(irrelevant=0):
    """Generate exhaustively all samples from the 7-segment problem.

    Parameters
    ----------
    irrelevant : int, optional (default=0)
        The number of irrelevant binary features to add. Since samples are
        generated exhaustively, this makes the size of the resulting dataset
        2^(irrelevant) times larger.

    Returns
    -------
    X, y
    """
    data = np.array([[0, 0, 1, 0, 0, 1, 0, 1],
                     [1, 0, 1, 1, 1, 0, 1, 2],
                     [1, 0, 1, 1, 0, 1, 1, 3],
                     [0, 1, 1, 1, 0, 1, 0, 4],
                     [1, 1, 0, 1, 0, 1, 1, 5],
                     [1, 1, 0, 1, 1, 1, 1, 6],
                     [1, 0, 1, 0, 0, 1, 0, 7],
                     [1, 1, 1, 1, 1, 1, 1, 8],
                     [1, 1, 1, 1, 0, 1, 1, 9],
                     [1, 1, 1, 0, 1, 1, 1, 0]])

    X, y = np.array(data[:, :7], dtype=np.bool), data[:, 7]

    if irrelevant > 0:
        X_ = []
        y_ = []

        for i in range(10):
            for s in itertools.product(range(2), repeat=irrelevant):
                X_.append(np.concatenate((X[i], s)))
                y_.append(i)

        X = np.array(X_, dtype=np.bool)
        y = np.array(y_)

    return X, y
def make_led_sample(n_samples=200, irrelevant=0, random_state=None):
    """Generate random samples from the 7-segment problem.

    Parameters
    ----------
    n_samples : int, optional (default=200)
        The number of samples to generate.

    irrelevant : int, optional (default=0)
        The number of irrelevant binary features to add.

    Returns
    -------
    X, y
    """

    random_state = check_random_state(random_state)

    data = np.array([[0, 0, 1, 0, 0, 1, 0, 1],
                     [1, 0, 1, 1, 1, 0, 1, 2],
                     [1, 0, 1, 1, 0, 1, 1, 3],
                     [0, 1, 1, 1, 0, 1, 0, 4],
                     [1, 1, 0, 1, 0, 1, 1, 5],
                     [1, 1, 0, 1, 1, 1, 1, 6],
                     [1, 0, 1, 0, 0, 1, 0, 7],
                     [1, 1, 1, 1, 1, 1, 1, 8],
                     [1, 1, 1, 1, 0, 1, 1, 9],
                     [1, 1, 1, 0, 1, 1, 1, 0]])

    data = data[random_state.randint(0, 10, n_samples)]
    X, y = np.array(data[:, :7],  dtype=np.bool), data[:, 7]

    if irrelevant > 0:
        X = np.hstack((X, random_state.rand(n_samples, irrelevant) > 0.5))

    return X, y

def dataset(name):
    """ This function returns a dataset given a name
    X: input 
    y: output
    t: nature of the output variable (useful if the use of results depends on it)
    """
    t = "unknown"
    if name =="boston":
        # regression (506x13feat)
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)
        t = "R"
        #X,y = shap.datasets.boston()
        #return X,y
    elif name == "iris":
        # classification (150x4featx3classes)
        from sklearn.datasets import load_iris
        data = load_iris()
        X = data.data
        y = data.target
        t = "C"
    elif name == "diabetes":
        # regression (442x10feat)
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(return_X_y=True)
        t = "R"
    elif name == "digits":
        # classification (1797x64featx10classes)
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        t = "C"
    elif name == "wine":
        # classification (178x13featuresx3classes)
        from sklearn.datasets import load_wine
        X, y = load_wine(return_X_y=True)
        t = "C"
    elif name == "breast_cancer":
        # classification (569x30featx2classes)
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        t = "C"
    elif name =="nhanesi":
        X,y = shap.datasets.nhanesi()
        t = "R"
    elif name == "segments":
        X,y = make_led()
        t = "C"
    elif name == "segments_sampled":
        X,y = make_led_sample()
        t = "C"
    elif name == "friedman1":
        from sklearn.datasets import make_friedman1
        X,y= make_friedman1(n_samples=500, random_state=0)
        print('Done')
        X = pd.DataFrame(X, columns=list(range(X.shape[1])))
        t = 'R'
    elif name == "friedman2":
        from sklearn.datasets import make_friedman2
        X,y= make_friedman2(random_state=0)
        t = 'R'
    elif name == 'linear':
        X, y, t = draw_linear_function()
    elif name == "linear2":
        importlib.reload(lreg)
        X,y,t = lreg.lf_dataset(nsamples=5000, with_vimp=False)
    elif name == 'friendman3':
        X, y, t = friedman_modified()
    else:
        raise ValueError("dataset `{}` not implemented".format(name))
    return X,y,t

from scipy.stats import norm

def friedman_modified():
    data, y, t = dataset('friedman1')
    data[0] = (data[0] + data[1])/2
    return data[[0,2,3,4,5,6,7,8,9]], y, t



def draw_linear_function():

    size = 1000
    X = np.ndarray(shape=(size, 5), dtype=float)
    loc = [20,13,4,15,7]
    scale = [6,7,8,10,5]
    for i in range(0,5):
        X[:,i] = norm.rvs(loc=loc[i], scale = scale[i], size=size)
    t = 'R'
    y = 50 * X[:,0] - 15 * X[:,1] + X[:,2]

    return X,y,t

from sklearn import preprocessing
def add_correlation(ranking1, values = [50, -15, 1,0,0], K = range(0,5) ):
    plt.figure(1)
    rank = preprocessing.normalize(np.array(values).reshape(1,-1), norm='l2', axis=1)
    rank = rank[0]
    coef1 = np.zeros(ranking1.shape[0])
    p1 = np.zeros(ranking1.shape[0])
    for i in range(ranking1.shape[0]):
        coef1[i], p1[i] = sc.spearmanr(ranking1[i, K], rank)

    x = list(range(0,ranking1.shape[0]))
    plt.plot(x, np.sort(coef1), label= 'Comparison with truth' + ", {} | {}".format(round(np.mean(coef1),3), round(np.median(coef1),3)))


import pydotplus
from sklearn.tree import export_graphviz
from io import StringIO
from sklearn import tree
from IPython.display import Image
from datetime import datetime
import graphviz

import pandas as pd


def save_dataset(name):
    X_, y_, t = dataset("friedman1")
    data = pd.DataFrame()
    data[list(range(X_.shape[1]))] = X_
    data['y'] = y_
    data.to_csv('datasets/' + name + '.csv', sep = ',', index=False)

    # dir = 'datasets/' + name + '.csv'
    # data = pd.read_csv(dir, low_memory=False, sep=",", encoding='utf8')
    # X = data.iloc[:, :-1]
    # y = data.iloc[:, -1]

    return data

def sample_feature_importance(mdiloc, mdaloc, SAABAS, SHAP, i):
    pprint("Local MDI", mdiloc[i, :])
    pprint("Local MDA", mdaloc[i, :])
    pprint("Saabas", SAABAS[i, :])
    pprint("SHAP", SHAP[i, :])

import scipy.stats as sc
import matplotlib.pyplot as plt

def spearman2_corr(mdiloc, mdaloc, title):
    # print("H0: two sets are uncorrelated")
    print("Feature correlation: " + title)
    for i in range(mdiloc.shape[1]):
        coef, p = sc.spearmanr(mdiloc[:,i], mdaloc[:,i])
        print("Feature ", i , ", Coef= ", coef,", p-val= ", p)

def spearman_corr(mdiloc, mdaloc):
    print("H0: two sets are uncorrelated")
    coef = np.zeros(mdiloc.shape[0])
    p = np.zeros(mdiloc.shape[0])
    for i in range(mdiloc.shape[0]):
        coef[i], p[i] = sc.spearmanr(mdiloc[i,:], mdaloc[i,:])
        # print("Coef= ", coef,", p-val= ", p)
    x = list(range(0, mdiloc.shape[0]))
    plt.plot(x,np.sort(coef), label="test")
    plt.legend()
    plt.axhline(y = -0.75, color='r', linestyle='-', linewidth=0.5)
    plt.axhline(y = 0.75, color='r', linestyle='-', linewidth=0.5)
    plt.title("Spearman correlation coefficient for all samples ranked")
    print("average: ", np.mean(coef))
    print("median ", np.median(coef))
    print("std ", np.std(coef))
    print('quantile 25 ', np.quantile(coef, 0.25))
    print('quantile 75 ', np.quantile(coef, 0.75))
    return coef, p


def spearman_corr_K_bis(mdiloc, mdaloc, K):
    print("H0: two sets are uncorrelated")
    coef = np.zeros(mdiloc.shape[0])
    p = np.zeros(mdiloc.shape[0])
    for i in range(mdiloc.shape[0]):
        coef[i], p[i] = sc.spearmanr(mdiloc[i,K], mdaloc[i,K])
    return coef, p

def spearman_corr_K(mdiloc, mdaloc, K):
    print("H0: two sets are uncorrelated")
    coef = np.zeros(mdiloc.shape[0])
    p = np.zeros(mdiloc.shape[0])
    for i in range(mdiloc.shape[0]):
        coef[i], p[i] = sc.spearmanr(mdiloc[i,K], mdaloc[i,K])
        # print("Coef= ", coef,", p-val= ", p)
    plt.plot(np.sort(coef))
    plt.axhline(y = -0.75, color='r', linestyle='-', linewidth=0.5)
    plt.axhline(y = 0.75, color='r', linestyle='-', linewidth=0.5)
    plt.title("Spearman correlation coefficient for all samples ranked K={}".format(K))
    print("average: ", np.mean(coef))
    print("median ", np.median(coef))
    print('quantile 25 ', np.quantile(coef, 0.25))
    print('quantile 75 ', np.quantile(coef, 0.75))
    return coef, p


def display_spearman_mdi(mdiloc, mdaloc, SAABAS, SHAP, global_mdi, method, order, K):
    coef1 = np.zeros(mdiloc.shape[0])
    coef2 = np.zeros(mdiloc.shape[0])
    coef3 = np.zeros(mdiloc.shape[0])
    coef4 = np.zeros(mdiloc.shape[0])
    p1 = np.zeros(mdiloc.shape[0])
    p2 = np.zeros(mdiloc.shape[0])
    p3 = np.zeros(mdiloc.shape[0])
    p4 = np.zeros(mdiloc.shape[0])

    for i in range(mdiloc.shape[0]):
        coef1[i], p1[i] = sc.spearmanr(mdiloc[i, K], mdaloc[i, K])
        coef2[i], p2[i] = sc.spearmanr(mdiloc[i, K], SHAP[i, K])
        coef3[i], p3[i] = sc.spearmanr(mdiloc[i, K], SAABAS[i, K])
        coef4[i], p4[i] = sc.spearmanr(mdiloc[i, K], global_mdi[K])


    # print("Coef= ", coef,", p-val= ", p)
    x = list(range(0,mdiloc.shape[0]))
    plt.figure()
    plt.plot(x, np.sort(coef1), label= order[0] + ", {} | {}".format(round(np.mean(coef1),3), round(np.median(coef1),3)))
    plt.plot(x, np.sort(coef2), label= order[1] + ", {} | {}".format(round(np.mean(coef2),3), round(np.median(coef2),3)))
    plt.plot(x, np.sort(coef3), label= order[2] + ", {} | {}".format(round(np.mean(coef3),3), round(np.median(coef3),3)))
    plt.plot(x, np.sort(coef4), label= 'Global MDI')
    plt.legend()
    plt.axhline(y=-0.75, color='r', linestyle='-', linewidth=0.5)
    plt.axhline(y=0.75, color='r', linestyle='-', linewidth=0.5)
    plt.ylim((-1, 1))
    plt.title(method + " : Spearman correlation coefficient for all samples ranked K={}".format(K))
    print("average: ", np.mean(coef1))
    print("median ", np.median(coef1))
    return

def display_spearman_mdi_class(mdiloc, mdaloc, SAABAS, SHAP, method, order, K):
    coef1 = np.zeros(mdiloc.shape[0])
    coef2 = np.zeros(mdiloc.shape[0])
    coef3 = np.zeros(mdiloc.shape[0])
    coef4 = np.zeros(mdiloc.shape[0])
    p1 = np.zeros(mdiloc.shape[0])
    p2 = np.zeros(mdiloc.shape[0])
    p3 = np.zeros(mdiloc.shape[0])
    p4 = np.zeros(mdiloc.shape[0])

    for i in range(mdiloc.shape[0]):
        coef1[i], p1[i] = sc.spearmanr(mdiloc[i, K], mdaloc[i, K])
        coef2[i], p2[i] = sc.spearmanr(mdiloc[i, K], SHAP[i, K])
        coef3[i], p3[i] = sc.spearmanr(mdiloc[i, K], SAABAS[i, K])

    # print("Coef= ", coef,", p-val= ", p)
    x = list(range(0,mdiloc.shape[0]))
    plt.figure()
    plt.plot(x, np.sort(coef1), label= order[0] + ", {} | {}".format(round(np.mean(coef1),3), round(np.median(coef1),3)))
    plt.plot(x, np.sort(coef2), label= order[1] + ", {} | {}".format(round(np.mean(coef2),3), round(np.median(coef2),3)))
    plt.plot(x, np.sort(coef3), label= order[2] + ", {} | {}".format(round(np.mean(coef3),3), round(np.median(coef3),3)))
    plt.legend()
    plt.axhline(y=-0.75, color='r', linestyle='-', linewidth=0.5)
    plt.axhline(y=0.75, color='r', linestyle='-', linewidth=0.5)
    plt.ylim((-1, 1))
    plt.title(method + " : Spearman correlation coefficient" + '\n' + "for all samples ranked K={}".format(K))
    print("average: ", np.mean(coef1))
    print("median ", np.median(coef1))
    return


def display_spearman_mdi_comparison(mdiloc, mdaloc, SAABAS, SHAP, global_mdi, method, order, K):
    coef1 = np.zeros(mdiloc.shape[0])
    coef2 = np.zeros(mdiloc.shape[0])
    coef3 = np.zeros(mdiloc.shape[0])
    coef4 = np.zeros(mdiloc.shape[0])
    p1 = np.zeros(mdiloc.shape[0])
    p2 = np.zeros(mdiloc.shape[0])
    p3 = np.zeros(mdiloc.shape[0])
    p4 = np.zeros(mdiloc.shape[0])

    mda_avg = np.average(mdaloc, axis=0)
    SAABAS_avg = np.average(SAABAS, axis=0)
    SHAP_avg = np.average(SHAP, axis=0)


    for i in range(mdiloc.shape[0]):
        coef1[i], p1[i] = sc.spearmanr(mdiloc[i, K], mda_avg[K])
        coef2[i], p2[i] = sc.spearmanr(mdiloc[i, K], SHAP_avg[K])
        coef3[i], p3[i] = sc.spearmanr(mdiloc[i, K], SAABAS_avg[K])
        coef4[i], p4[i] = sc.spearmanr(mdiloc[i, K], global_mdi[K])


    # print("Coef= ", coef,", p-val= ", p)
    x = list(range(0,mdiloc.shape[0]))
    plt.figure()
    plt.plot(x, np.sort(coef1), label= order[0] + ", {} | {}".format(round(np.mean(coef1),3), round(np.median(coef1),3)))
    plt.plot(x, np.sort(coef2), label= order[1] + ", {} | {}".format(round(np.mean(coef2),3), round(np.median(coef2),3)))
    plt.plot(x, np.sort(coef3), label= order[2] + ", {} | {}".format(round(np.mean(coef3),3), round(np.median(coef3),3)))
    plt.plot(x, np.sort(coef4), label= 'Global MDI')
    plt.legend()
    plt.axhline(y=-0.75, color='r', linestyle='-', linewidth=0.5)
    plt.axhline(y=0.75, color='r', linestyle='-', linewidth=0.5)
    plt.ylim((-1, 1))
    plt.title(method + " : Spearman correlation coefficient wrt mean" + "\n" +"for all samples ranked K={}".format(K))
    print("average: ", np.mean(coef1))
    print("median ", np.median(coef1))
    return



def display_spearman_mda(mdiloc, mdaloc,SAABAS, SHAP, method, order, K, frequency_control= "", dataset = "", values=[50, 15, 1,0,0], do =False):


    coef0 = np.zeros(mdaloc.shape[0])
    p0 = np.zeros(mdaloc.shape[0])
    if do == True:
        rank = preprocessing.normalize(np.array(values).reshape(1, -1), norm='l2', axis=1)
        rank = rank[0]
        for i in range(mdaloc.shape[0]):
            coef0[i], p0[i] = sc.spearmanr(mdaloc[i, K], rank)

    coef5 = np.zeros(mdaloc.shape[0])
    p5 = np.zeros(mdaloc.shape[0])
    if do == True:
        rank = preprocessing.normalize(np.array(values).reshape(1, -1), norm='l2', axis=1)
        rank = rank[0]
        for i in range(mdaloc.shape[0]):
            coef5[i], p5[i] = sc.spearmanr(SHAP[i, K], rank)


    coef1 = np.zeros(mdiloc.shape[0])
    coef2 = np.zeros(mdiloc.shape[0])
    coef3 = np.zeros(mdiloc.shape[0])
    p1 = np.zeros(mdiloc.shape[0])
    p2 = np.zeros(mdiloc.shape[0])
    p3 = np.zeros(mdiloc.shape[0])

    for i in range(mdiloc.shape[0]):
        coef1[i], p1[i] = sc.spearmanr(mdaloc[i, K], mdiloc[i, K])
        coef2[i], p2[i] = sc.spearmanr(mdaloc[i, K], SHAP[i, K])
        coef3[i], p3[i] = sc.spearmanr(mdaloc[i, K], SAABAS[i, K])

    # print("Coef= ", coef,", p-val= ", p)
    x = list(range(0,mdiloc.shape[0]))
    plt.figure()
    plt.plot(x, np.sort(coef1), label= order[0] + ", {} | {}".format(round(np.mean(coef1),3), round(np.median(coef1),3)))
    plt.plot(x, np.sort(coef2), label= order[1] + ", {} | {}".format(round(np.mean(coef2),3), round(np.median(coef2),3)))
    plt.plot(x, np.sort(coef3), label= order[2] + ", {} | {}".format(round(np.mean(coef3),3), round(np.median(coef3),3)))
    plt.plot(x, np.sort(coef0), label='Comparison with truth' + ", {} | {}".format(round(np.mean(coef0), 3),round(np.median(coef0), 3)))
    plt.plot(x, np.sort(coef5), label='Comparison of Shap with truth' + ", {} | {}".format(round(np.mean(coef5), 3),round(np.median(coef5), 3)))

    plt.legend()
    plt.axhline(y=-0.75, color='r', linestyle='-', linewidth=0.5)
    plt.axhline(y=0.75, color='r', linestyle='-', linewidth=0.5)
    plt.ylim((-1, 1))
    plt.title(method + " : Spearman correlation coefficient" + '\n' + "for all samples ranked K={}".format(K) +
              " " + frequency_control + " " + dataset)
    print("average: ", np.mean(coef1))
    print("median ", np.median(coef1))

    return

def display_spearman_mda_comparison(mdiloc, mdaloc, SAABAS, SHAP, method, order, K, frequency_control= "", dataset = "", values=[50, 15, 1,0,0], do =False):

    coef0 = np.zeros(mdaloc.shape[0])
    p0 = np.zeros(mdaloc.shape[0])
    if do == True:
        rank = preprocessing.normalize(np.array(values).reshape(1, -1), norm='l2', axis=1)
        rank = rank[0]
        for i in range(mdaloc.shape[0]):
            coef0[i], p0[i] = sc.spearmanr(mdaloc[i, K], rank)
    
    coef1 = np.zeros(mdiloc.shape[0])
    coef2 = np.zeros(mdiloc.shape[0])
    coef3 = np.zeros(mdiloc.shape[0])
    p1 = np.zeros(mdiloc.shape[0])
    p2 = np.zeros(mdiloc.shape[0])
    p3 = np.zeros(mdiloc.shape[0])

    mdi_avg = np.average(mdiloc, axis=0)
    SAABAS_avg = np.average(SAABAS, axis=0)
    SHAP_avg = np.average(SHAP, axis=0)

    for i in range(mdiloc.shape[0]):
        coef1[i], p1[i] = sc.spearmanr(mdaloc[i, K], mdi_avg[K])
        coef2[i], p2[i] = sc.spearmanr(mdaloc[i, K], SHAP_avg[K])
        coef3[i], p3[i] = sc.spearmanr(mdaloc[i, K], SAABAS_avg[K])

    # print("Coef= ", coef,", p-val= ", p)
    x = list(range(0,mdiloc.shape[0]))
    plt.figure()
    plt.plot(x, np.sort(coef1), label= order[0] + ", {} | {}".format(round(np.mean(coef1),3), round(np.median(coef1),3)))
    plt.plot(x, np.sort(coef2), label= order[1] + ", {} | {}".format(round(np.mean(coef2),3), round(np.median(coef2),3)))
    plt.plot(x, np.sort(coef3), label= order[2] + ", {} | {}".format(round(np.mean(coef3),3), round(np.median(coef3),3)))
    plt.plot(x, np.sort(coef0), label='Comparison with truth' + ", {} | {}".format(round(np.mean(coef0), 3),round(np.median(coef0), 3)))

    plt.legend()
    plt.axhline(y=-0.75, color='r', linestyle='-', linewidth=0.5)
    plt.axhline(y=0.75, color='r', linestyle='-', linewidth=0.5)
    plt.ylim((-1, 1))
    plt.title(method + " : Spearman correlation coefficient wrt mean" + '\n' + "for all samples ranked K={}".format(K) +
              " " + frequency_control + " " + dataset)
    print("average: ", np.mean(coef1))
    print("median ", np.median(coef1))

    return

def display_corr(mdiloc, mdaloc):

    for i in range(mdiloc.shape[1]):
        plt.figure(i)
        plt.scatter(mdiloc[:,i], mdaloc[:,i])
        plt.title('Feature T={}'.format(i))
        plt.ylabel('mdaloc')
        plt.xlabel('mdiloc')
        plt.show

def pearson_corr(mdiloc, mdaloc):
    print("H0: two sets are uncorrelated")
    coef = np.zeros(mdiloc.shape[0])
    p = np.zeros(mdiloc.shape[0])
    for i in range(mdiloc.shape[0]):
        coef[i], p[i] = sc.pearsonr(mdiloc[i,:], mdaloc[i,:])
        # print("Coef= ", coef,", p-val= ", p)
    plt.plot(np.sort(coef))
    plt.axhline(y = -0.75, color='r', linestyle='-', linewidth=0.5)
    plt.axhline(y = 0.75, color='r', linestyle='-', linewidth=0.5)
    plt.title("Pearson correlation coefficient for all samples ranked")
    print("average: ", np.mean(coef))
    print("median ", np.median(coef))
    print('quantile 25 ', np.quantile(coef, 0.25))
    print('quantile 75 ', np.quantile(coef, 0.75))
    return coef, p

def ranking_compare(mda, rank2, ranking=True, view = False):
    if ranking == True:
        rank2_avg = np.average(rank2, axis=0)
    coef = np.zeros(mda.shape[0])
    p = np.zeros(mda.shape[0])
    for i in range(mda.shape[0]):
        coef[i], p[i] = sc.spearmanr(mda[i, :], rank2_avg)

    if view == True:
        plt.figure()
        plt.plot(np.sort(coef))
        plt.axhline(y=-0.75, color='r', linestyle='-', linewidth=0.5)
        plt.axhline(y=0.75, color='r', linestyle='-', linewidth=0.5)
        plt.title("Spearman correlation coefficient for all samples ranked")
        print("average: ", np.mean(coef))
        print("median ", np.median(coef))
        print('quantile 25 ', np.quantile(coef, 0.25))
        print('quantile 75 ', np.quantile(coef, 0.75))
    return coef


##########

def jaccard_formula(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

# return the index of the #size most important features
def feature_ranking(fimp, size):
    return np.argsort(fimp)[::-1][0:size]

# To use
def shared_ranking_jaccard(similarity, size):
    return similarity*2*size/(1+similarity)

# To use
def jaccard_index(rank1, rank2):

    nsamples = rank1.shape[0]
    nfeatures = rank1.shape[1]
    plt.figure()
    for i in list([2,4,5,7]):
    # for i in range(1,nfeatures):
        jaccard = np.zeros(nsamples)
        for j in range(0, nsamples):
            jaccard[j] = jaccard_formula(feature_ranking(rank1[j], i), feature_ranking(rank2[j], i))
        plt.plot(np.sort(jaccard),'.-', label= "Jaccard Index for K = {}".format(i))
        plt.legend()
        plt.title("Jaccard Similarity Index")
    return

def jaccard_index2(rank1, rank2):

    nsamples = rank1.shape[0]
    nfeatures = rank1.shape[1]
    plt.figure()
    for i in list([5]):
    # for i in range(1,nfeatures):
        jaccard = np.zeros(nsamples)
        for j in range(0, nsamples):
            jaccard[j] = jaccard_formula(feature_ranking(rank1[j], i), feature_ranking(rank2[j], i))
        plt.plot(np.sort(jaccard),'.-', label= "Jaccard Index for K = {}".format(i))
        plt.legend()
        plt.title("Jaccard Similarity Index")
        print('i = '.format(i))
        print('avg jaccard: ', np.mean(jaccard))
        print('median jaccard: ', np.median(jaccard))
        print('std jaccard: ', np.std(jaccard))
    return


#########
import collections as col

# rank_stat([mdiloc[0],mdaloc[0], SAABAS[0], SHAP[0]], ['mdi','mda', "SAABAS", "SHAP"])
# rank_stat([mdiloc[0],mdaloc[0], SAABAS[0]], ['mdi','mda', "SAABAS"])
# rank([mdiloc, mdaloc, SAABAS, SHAP], types=['mdiloc', 'mdaloc', 'SAABAS', "SHAP"],)
def rank_stat(rankings, types, true_var=5):

    n_features = rankings[0].shape[1]
    n_samples = rankings[0].shape[0]
    rank_freq = []
    rank_avg = []
    n_types = len(types)
    for i in rankings:
        rank_avg.append(np.argsort(np.argsort(i)[:,::-1]))
        rank = np.argsort(abs(i))
        rank = rank[:,::-1][:, 0]
        rank = col.Counter(rank)
        rank_freq.append(rank)
    for j in range(n_features):
        if j == true_var:
            print("**** Noise Variables ****")
        tmp = 'Feature ' + str(j) + ' : '
        for k in range(0, n_types):
            tmp += types[k] + ' ' + "{:3.2f}".format(round(rank_freq[k][j]/n_samples, 2))
            tmp += '- avg rank: ' + "{:3.2f}".format(rank_avg[k][:,j].mean()) +  ' | '
        print(tmp.format())

    tmp = 'Likelihood to rank 1st a True-variable: '
    for i in range(0,n_types):
        sumT = 0
        for k in range(0,true_var):
            sumT += rank_freq[i][k]
        tmp += types[i] + ' ' + "{:3.2f}".format(sumT) + ' | '
    print(tmp)

    # Proportion of rankins whose most important variable are the true_var
    true_ranks = []
    tmp = 'Percent of rankings that correspond to the True variables: '
    l = 0
    for i in rankings:
        rank = np.argsort(abs(i))
        rank = np.sum(rank[:,::-1][:, 0:5], axis=1)
        val = sum(rank==15)
        tmp += types[l] + ' ' + "{:3.2f}".format(val/n_samples) + ' | '
        l +=1
    print(tmp)
    return
#######################################


def CL_ranking(vimp):
    nsamples, nfeatures, c = vimp.shape
    forecast = np.zeros((nsamples, nfeatures))
    for i in range(0,nsamples):
        for j in range(0,nfeatures):
            forecast[i,j]  = np.argmax(vimp[i,j,:])
    return forecast
# CL_ranking(mdalocC)

def check_ranking(forecast, y):
    nsamples, nfeatures = forecast.shape
    results = np.zeros(nsamples)
    for i in range(0,nsamples):
        if sum(forecast[i])/nfeatures != y[i]:
            results[i] = 1
    if sum(results) != 0:
        print("One forecast is different than ground truth")

    return results


def Classification_error(vimp, y):
    nsamples, nfeatures, c = vimp.shape
    vimp2 = np.zeros((nsamples, nfeatures))
    for i in range(0,nsamples):
        for j in range(0,nfeatures):
            #vimp2[i,j] = sum(vimp[i,j,:]) - vimp[i,j,y[i]]
            vimp2[i, j] = 1 - vimp[i, j, y[i]]
    return vimp2


def Saabas_imp(contributions, y):
    nsamples, nfeatures, c = contributions.shape
    vimp2 = np.zeros((nsamples, nfeatures))
    for i in range(0,nsamples):
        for j in range(0,nfeatures):
            vimp2[i, j] = contributions[i, j, y[i]]
    return vimp2


def Shap_imp(contributions, y):
    c, nsamples, nfeatures = contributions.shape
    vimp2 = np.zeros((nsamples, nfeatures))
    for i in range(0,nsamples):
        for j in range(0,nfeatures):
            vimp2[i, j] = contributions[y[i], i, j]
    return vimp2
# mdalocC2 = Classification_error(mdalocC, y)

