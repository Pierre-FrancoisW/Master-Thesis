import scipy.stats as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lin_reg as lreg
import importlib

def expressivity(methods= [], names = []):
    x = list(range(0, methods[0].shape[0]))
    val = len(methods)
    print(val)
    plt.figure()
    for i in range(0,val):
        tmp = np.max(methods[i], axis=1) - np.min(methods[i], axis=1)
        plt.plot(x,tmp,label=names[i] + ", {} ".format(round(np.mean(tmp), 3)))
    plt.legend()
    plt.title('Expresivity: max - min')
    return

def correlation_summary(method1, method2, return_coef =  False):
    # K is the range of variable to study
    coef = np.zeros(method1.shape[0])
    p = np.zeros(method1.shape[0])
    for i in range(method1.shape[0]):
        coef[i], p[i] = sc.spearmanr(method1[i, :], method2[i, :])

    print("average: ", np.mean(coef))
    print("Std: ", np.std(coef))
    print("median: ", np.median(coef))
    print('quantile 25: ', np.quantile(coef, 0.25))
    print('quantile 75: ', np.quantile(coef, 0.75))
    if return_coef:
        return coef
    else:
        return


def spearman_corr_disp(method1, method2, K, legend = "", return_val = False):
    plt.figure()
    print("H0: two sets are uncorrelated")
    coef = np.zeros(method1.shape[0])
    p = np.zeros(method1.shape[0])
    for i in range(method1.shape[0]):
        coef[i], p[i] = sc.spearmanr(method1[i,K], method2[i,K])
        # print("Coef= ", coef,", p-val= ", p)
    plt.plot(np.sort(coef), label= legend)
    plt.axhline(y = -0.75, color='r', linestyle='-', linewidth=0.5)
    plt.axhline(y = 0.75, color='r', linestyle='-', linewidth=0.5)
    plt.title("Spearman correlation coefficient for all samples ranked K={}".format(K))
    print("average: ", np.mean(coef))
    print("median ", np.median(coef))
    print('quantile 25 ', np.quantile(coef, 0.25))
    print('quantile 75 ', np.quantile(coef, 0.75))

    if return_val:
        return coef, p
    else:
        return

#  Display the example, value, abs_sum, and the importance associated by each methods
#  with their local ranking
def lr_method_comparisons(sample,index, methods=[], names = []):

    importlib.reload(lreg)
    weight = lreg.weight_lin_reg
    b = lreg.b_lin_reg
    value, abs_sum = lreg.linear_function(sample)
    size = len(sample)
    #print("{0:3d} | {1:3d} | {2:3d} | {3:4.2f} | {4:3.3f} | {5:3.3f} | {6:3.3f} |".format(val[0], val[1], val[2], tmp,abs(abs(val[0] * weight[0])/tmp),abs(abs(val[1] * weight[1])/tmp),abs(abs(val[2] * weight[2])/tmp) ))
    tmp = ''
    for i in range(0,size):
        tmp += 'x{}: '.format(i)
        tmp += '{0:1.3f} | '.format(sample[i])
    tmp += ' weights: '
    for i in range(0,size):
        tmp += '{0:3.1f} , '.format(weight[i])
    tmp += '| function value: {0:5.3f}, '.format(value)
    tmp += ' abs sum of elements: {0:5.3f}'.format(abs_sum)
    print("sample:  " + tmp)


    for i in range(0, len(names)):
        tmp = '{:<8}'.format(names[i]) + '|'
        ranking = methods[i][index]
        for j in range(0, size):
            tmp += 'x{}: '.format(j)
            tmp += '{0:1.3f} | '.format(ranking[j])

        tmp + ' ranking: '
        vimp_index = np.argsort(ranking)[::-1]
        for j in range(0, size):
            tmp += 'x{} , '.format(vimp_index[j])
        print(tmp)

    tmp = '{}'.format('Weighted values') + ' | '
    for i in range(0, size):
        tmp += '{0:3.3f} |'.format((weight[i] * sample[i]))
    print(tmp)


def ranking_compare2(mda, rank2, ranking=False, view = False, method = ''):

    rank2_avg = rank2
    coef = np.zeros(mda.shape[0])
    p = np.zeros(mda.shape[0])
    for i in range(mda.shape[0]):
        coef[i], p[i] = sc.spearmanr(mda[i, :], rank2_avg[i, :])

    # Show proportion of rankings that share the same feature at all positions
    nsamples = mda.shape[0]
    nfeatures = mda.shape[1]
    for i in range(0, nfeatures):
        bool = np.argsort(mda)[:, nfeatures-1 - i] ==  np.argsort(rank2_avg)[:, nfeatures-1 - i]
        print("Proportion of same feature ranked {0:} : {1:}".format(i,sum(bool)/nsamples))



    if view == True:
        plt.figure()
        plt.plot(np.sort(coef))
        plt.axhline(y=-0.75, color='r', linestyle='-', linewidth=0.5)
        plt.axhline(y=0.75, color='r', linestyle='-', linewidth=0.5)
        x = list(range(0, mda.shape[0]))
        plt.plot(x, np.sort(coef),label = method + ", {} | {}".format(round(np.mean(coef), 3), round(np.median(coef), 3)))
        plt.legend()
        plt.title("Spearman correlation "+ method)
        print("average: ", np.mean(coef))
        print("median ", np.median(coef))
        print('std ', np.std(coef))
        print('quantile 25 ', np.quantile(coef, 0.25))
        print('quantile 75 ', np.quantile(coef, 0.75))
    return


def build_global_ranking(ranking):
    weight = lreg.weight_lin_reg
    return np.tile(np.abs(weight), (ranking.shape[0],1))

def build_ranking(ranking, Ranking2):
    size = Ranking2.shape[0]
    return np.tile(np.abs(ranking), (size, 1))




