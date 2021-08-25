from sklearn.utils import check_random_state
import numpy as np
import matplotlib.pyplot as plt
import importlib
import time

import utils
from utils import pprint

from sklearn.model_selection import train_test_split



def regression_task(name= 'friedman1', n_estimators=1000, frequency_control= 'weighted', K=1, disp=True, Normalized = True, Uniform = False, DoSplit = True, Error_avg = True, lRegTT = True, norm = 'l1'):

    import utils as us
    from sklearn.ensemble import ExtraTreesRegressor
    import LocalMDI_cy
    import local_mda2
    from treeinterpreter import treeinterpreter as Ti
    import shap
    import lin_reg as lreg
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    importlib.reload(local_mda2)
    importlib.reload(us)
    importlib.reload(lreg)

    ## Make a dataset
    X_, y, t = us.dataset(name)
    # Get the size of dataset

    # Required change of input type
    X = np.asarray(X_, dtype=np.float32)
    Y = np.asarray(y, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.20, random_state=1)

    # X_train = X
    # y_train = y
    # X_test = X
    # X_train = X_train[:, range(0,5)]
    # X_test = X_test[:, range(0,5)]

    n_samples, n_features = X_test.shape


    # Create a Totally randomized trees T=1000, K=1

    if t == "R":
        model = ExtraTreesRegressor(n_estimators=n_estimators, max_features=K, max_depth=None, min_samples_split=2,
                                    random_state=0, verbose=0, n_jobs=-1)
    else:
        raise ValueError('T = {}?'.format(t))

    # Learn the model
    model.fit(X_train, y_train)

    # Get the predictions of the model
    predictions = model.predict(X_test)
    print("Mean square error:  ", mean_squared_error(y_test, predictions))
    print("Mean absolute error", mean_absolute_error(y_test, predictions))
    print("y range:", min(y_test), "to", max(y_test), "avg and median:", np.mean(y_test)," ", np.median(y_test))

    ## Compute feature importances
    X = X_test

    # Local MDI
    mdiloc = LocalMDI_cy.compute_mdi_local_ens(model, X)
    print("shape mdiloc: {}".format(mdiloc.shape))

    # Local MDA
    print("Doing MDA")
    start = time.time()
    mdaloc = local_mda2.compute_mda_local_ens(model, X, 'classic', frequency_control, uniform=Uniform, DoSplit=DoSplit, Error_avg = Error_avg)
    end = time.time()
    print(end - start)
    print("shape mdaloc: {}".format(mdaloc.shape))

    # Saabas and (Tree)Shap
    if t == "R":
        # Saabas
        print("Doing SAABAS...")
        prediction, bias, contributions = Ti.predict(model, X)
        SAABAS = contributions

        # SHAP
        print("Doing SHAP...")
        shap_values = shap.TreeExplainer(model).shap_values(X)
        SHAP = shap_values

    else:
        raise ValueError("{} not known".format(t))

    global_mdi = model.feature_importances_

    if Normalized == True:
        print("Normalize")
        mdaloc = normalize_rankings([mdaloc], norm=norm)[0]
        mdiloc = normalize_rankings([mdiloc], norm=norm)[0]
        SAABAS = normalize_rankings([SAABAS], norm)[0]
        SHAP = normalize_rankings([SHAP], norm)[0]


    if disp == True:
        print("Local importances for the first sample")
        pprint("Local MDI", mdiloc[0, :])
        pprint("Local MDA", mdaloc[0, :])
        pprint("Saabas", SAABAS[0, :])
        pprint("SHAP", SHAP[0, :])
        pprint("Global MDI", global_mdi)

        doit = False
        valuesss = [10, 50, 10,5, 0,0,0,0, 0]
        us.display_spearman_mda(mdiloc, mdaloc, abs(SAABAS), abs(SHAP), 'MDA', ["MDI", "SHAP", "SAABAS"],
                                list(range(0, mdiloc.shape[1])), frequency_control, name, do=doit, values=valuesss)
        # us.display_spearman_mdi(mdiloc, mdaloc, abs(SAABAS), abs(SHAP), global_mdi,  "MDI", ["MDA", "SAABAS", "SHAP"], list(range(0, mdiloc.shape[1])))
        #
        #
        # us.display_spearman_mda_comparison(mdiloc, mdaloc, abs(SAABAS), abs(SHAP), 'MDA', ["MDI", "SAABAS", "SHAP"],
        #                     list(range(0,mdiloc.shape[1])), frequency_control, name, do=doit, values=valuesss)
        # us.display_spearman_mdi_comparison(mdiloc, mdaloc, abs(SAABAS), abs(SHAP), global_mdi,  "MDI", ["MDA", "SAABAS", "SHAP"], list(range(0, mdiloc.shape[1])))

        us.spearman2_corr(mdiloc, mdaloc, " mdaloc and mdiloc")
        us.spearman2_corr(mdaloc, abs(SAABAS), " mdaloc and SAABAS")
        us.spearman2_corr(mdiloc, abs(SAABAS), " mdiloc and SAABAS")

    print("***Finished***")
    if lRegTT:
        X2,y2,t2, Vimp2 = lreg.lf_dataset(150, True)
        return X, model, mdiloc, mdaloc, abs(SAABAS), abs(SHAP), abs(Vimp2)
    else:
        return X, model, mdiloc, mdaloc, abs(SAABAS), abs(SHAP)
# MODELR, XR, mdilocR, mdalocR, saabasR, shapR = regression_task('friedman1', 500, 'n', K=1, disp=True, Normalized=True, Uniform=False)
# mdiloc, mdaloc, SAABAS, SHAP = make_measures(name= 'friedman1', n_estimators=[300, 500, 1000], frequency_control= 'n', K=1, disp=True, Normalized= True)
# MODELR, XR, mdilocR, mdalocR2, saabasR, shapR = regression_task('friedman1', 500, 'n', K=1, disp=True, Normalized=True, Uniform=False, DoSplit = False, Error_avg = False)

from sklearn import preprocessing
def normalize_rankings(ranking_batch, norm='l2'):
    size = len(ranking_batch)
    for rank in range(0,size):
        ranking_batch[rank] = preprocessing.normalize(ranking_batch[rank], norm=norm, axis=1)
    return ranking_batch


def make_measures(name= 'friedman1', n_estimators=[300, 400, 1000], frequency_control= 'weighted', K=1, disp=False, Normalized = True):
    MDILOC = []
    MDALOC = []
    SAABAS = []
    SHAP = []

    for i in n_estimators:
        _, _, mdiloc, mdaloc, saabas, shap = regression_task(name, i, frequency_control, K, disp=disp, Normalized=Normalized)
        MDILOC.append(mdiloc)
        MDALOC.append(mdaloc)
        SAABAS.append(saabas)
        SHAP.append(shap)
        print("Finished with n_estimators = {}".format(i))
    return MDILOC, MDALOC, SAABAS, SHAP

def make_measure_withK(name= 'friedman1', n_estimators=[500], frequency_control= 'weighted', disp=False, Normalized = True):
    MDILOC1, MDALOC1, SAABAS1, SHAP1 = make_measures(name= name, n_estimators=n_estimators, frequency_control= frequency_control, K=1, disp=disp, Normalized = Normalized)
    MDILOC2, MDALOC2, SAABAS2, SHAP2 = make_measures(name= name, n_estimators=n_estimators, frequency_control= frequency_control, K=2, disp=disp, Normalized = Normalized)
    MDILOC3, MDALOC3, SAABAS3, SHAP3 = make_measures(name= name, n_estimators=n_estimators, frequency_control= frequency_control, K=3, disp=disp, Normalized = Normalized)
    import utils as us
    plt.figure()
    avg1 = np.zeros((3, 1))
    avg2 = np.zeros((3, 1))
    avg3 = np.zeros((3, 1))
    K = np.array([1,2,3])

    Range = range(0,MDILOC1[0].shape[1])

    cf1, p1 = us.spearman_corr_K_bis(MDALOC1[0],MDILOC1[0] , Range)
    avg1[0] = cf1.mean()
    cf2, p2 = us.spearman_corr_K_bis(MDALOC1[0],SAABAS1[0] , Range)
    avg2[0] = cf2.mean()
    cf3, p3 = us.spearman_corr_K_bis(MDALOC1[0], SHAP1[0], Range)
    avg3[0] = cf3.mean()

    cf1, p1 = us.spearman_corr_K_bis(MDALOC2[0], MDILOC2[0], Range)
    avg1[1] = cf1.mean()
    cf2, p2 = us.spearman_corr_K_bis(MDALOC2[0], SAABAS2[0], Range)
    avg2[1] = cf2.mean()
    cf3, p3 = us.spearman_corr_K_bis(MDALOC2[0], SHAP2[0], Range)
    avg3[1] = cf3.mean()

    cf1, p1 = us.spearman_corr_K_bis(MDALOC3[0], MDILOC3[0], Range)
    avg1[2] = cf1.mean()
    cf2, p2 = us.spearman_corr_K_bis(MDALOC3[0], SAABAS3[0], Range)
    avg2[2] = cf2.mean()
    cf3, p3 = us.spearman_corr_K_bis(MDALOC3[0], SHAP3[0], Range)
    avg3[2] = cf3.mean()
    plt.plot(K, avg1, '*', label="MDI-MDA comparison | {}, {}, {}".format(avg1[0], avg1[1], avg1[2]), color = 'b')
    plt.plot(K, avg2, '*', label="MDA-SAABAS comparison | {}, {}, {}".format(avg2[0], avg2[1], avg2[2]), color='g')
    plt.plot(K, avg3, '*', label="MDA-SHAP comparison | {}, {}, {}".format(avg3[0], avg3[1], avg3[2]), color='r')
    plt.legend()
    plt.title("Average Correlation for K=1,2,3 and {} estimators".format(n_estimators[0]))
    plt.ylim((-1, 1))


def classification_task(name="iris", n_estimators=1000, max_features=1):
    import local_mda_class
    import utils as us
    from sklearn.ensemble import ExtraTreesClassifier
    import LocalMDI_cy
    import local_mda2
    from treeinterpreter import treeinterpreter as Ti
    import shap
    import tmp
    importlib.reload(tmp)

    importlib.reload(local_mda_class)
    importlib.reload(utils)

    ## Make a dataset
    X_, y, t = us.dataset(name)

    # Get the size of dataset

    # Required change of input type
    X = np.asarray(X_, dtype=np.float32)
    Y = np.asarray(y, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    print(X.shape)

    # Create a totally randomized FTRee
    K = 1
    if t == 'C':
        modelC= ExtraTreesClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=None, min_samples_split=2,
                                 criterion='entropy', random_state=0, verbose=0, n_jobs=-1)
    else:
        raise ValueError('T = {}?'.format(t))

    modelC.fit(X_train, y_train)
    predictions = modelC.predict(X_test)
    X = X_test
    n_samples, n_features = X_test.shape
    nclass = modelC.classes_.size


    # Local MDI
    mdiloc_C = LocalMDI_cy.compute_mdi_local_ens(modelC, X)
    print("shape mdiloc: {}".format(mdiloc_C.shape))

    # Local MDA
    mdaloc = tmp.compute_mda_local_ens_C(modelC, X, uniform = False)
    print("shape mdaloc: {}".format(mdaloc.shape))

    # Saabas and (Tree)Shap
    # The way to compute this will be different for classification and regression
    if t == "C":
        # In classification, Saabas & SHAP computes importance for each value of the output
        # Get the number of output values = number of classes
        n_classes = len(np.unique(y))
        print(n_classes)
        print("Doing SAABAS...")
        prediction, bias, contributions = Ti.predict(modelC, X)
        # print(contributions)
        # print(prediction)
        print("Doing SHAP...")
        shap_values = shap.TreeExplainer(modelC).shap_values(X)
        # Transform in order to have importances for the predicted class
        SAABAS_C = np.zeros((n_samples, n_features))
        SHAP_C = np.zeros((n_samples, n_features))
        SHAPVAL = np.zeros((n_classes, n_samples, n_features))
        y_values = np.unique(prediction)

        for j in range(len(np.unique(y))):
            SHAPVAL[j, :, :] = shap_values[j]
        for j in range(n_samples):
            #SAABAS_C[j, :] = SAABAS_C[j, :] + contributions[j, :, prediction[j] == 1]
            a = 1
            # SHAP_C[j, :] = SHAP_C[j, :] + SHAPVAL[prediction[j] == 1, j, :]
    else:
        raise ValueError('T = {}?'.format(t))

    print("Local importances for the first sample")
    pprint("Local MDI", mdiloc_C[0, :])
    print("Local MDA", mdaloc[0,:, :])
    # pprint("Saabas", SAABAS_C[0, :])
    # pprint("SHAP", SHAP_C[0, :])

    test = mdaloc
    mdaloc = us.Classification_error(mdaloc, y_test)
    SAABAS = us.Saabas_imp(contributions, y_test)
    SHAP = us.Shap_imp(SHAPVAL, y_test)
    SHAP = abs(SHAP)
    SAABAS = abs(SAABAS)
    SHAP = normalize_rankings([SHAP], 'l1')[0]
    SAABAS = normalize_rankings([SAABAS], 'l1')[0]
    mdaloc = normalize_rankings([mdaloc], 'l1')[0]
    mdiloc_C = normalize_rankings([mdiloc_C], 'l1')[0]



    return modelC, mdaloc, mdiloc_C, SAABAS, SHAP, X_test, y_test, predictions,prediction, contributions, test


def test_classification(name = 'iris',test = 'tmp', n_estimators=1000, max_features=1, uniform = False):
    from sklearn.ensemble import ExtraTreesClassifier
    import local_mda_class
    import tmp
    import utils as us
    importlib.reload(local_mda_class)
    importlib.reload(tmp)

    X_, y, t = us.dataset(name)
    X = np.asarray(X_, dtype=np.float32)
    Y = np.asarray(y, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    modelC = ExtraTreesClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=None,
                                  min_samples_split=2,
                                  criterion='entropy', random_state=0, verbose=0, n_jobs=-1)
    modelC.fit(X_train, y_train)
    predictions = modelC.predict(X_test)
    X = X_test
    n_samples, n_features = X_test.shape
    if test == 'tmp':
        mdaloc = tmp.compute_mda_local_ens_C(modelC, X, uniform)
    else:
        mdaloc = local_mda_class.compute_mda_local_ens_C(modelC, X)
    test = mdaloc
    mdalocV1 = us.Classification_error(mdaloc, y_test)
    mdalocV2 = tmp.Classification_vimp(mdaloc, modelC.predict_proba(X), y_test)
    mdalocV2 = abs(mdalocV2)
    mdalocV1 = normalize_rankings([mdalocV1])[0]
    mdalocV2 = normalize_rankings([mdalocV2])[0]

    return mdalocV1, mdalocV2

# modelC, mdaloc_C, mdiloc_C = classification_task(name="iris", n_estimators=10)
# model, mdiloc, mdaloc, SAABAS, SHAP = regression_task(name= 'friedman', n_estimators=10)


# display_spearman_mda(Bmdiloc, Bmdaloc, B_SHAP, B_SAABAS, "MDA_prob", ["MDI", "SHAP", "SAABAS"], list(range(0, 12)))
# display_spearman_mda(mdiloc, mdaloc, SHAP, SAABAS, "MDA", ["MDI", "SHAP", "SAABAS"], list(range(0,9)))
# display_spearman_mdi(mdiloc, mdaloc, SHAP, SAABAS, global_mdi,  "MDI", ["MDA", "SHAP", "SAABAS"], list(range(0,5)))
# display_spearman_mda(mdiloc, mdaloc2, SHAP, SAABAS, "MDA_prob", ["MDI", "SHAP", "SAABAS"], list(range(0,9)))





