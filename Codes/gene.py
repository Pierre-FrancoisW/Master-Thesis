import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
import pickle
from joblib import dump, load

import os
from datetime import datetime

import shap
from treeinterpreter import treeinterpreter as Ti
from main_local import normalize_rankings
import LocalMDI_cy
import local_mda2
import utils as us
import importlib

expression = 0
regulator_expression = 0
regulators = 0
regulators_network = 0
nsamples = 0
ngenes = 0
genes = 0
regulator_genes = 0
regulator_genes_id = []
nregulator_genes = 0

id_toGene = {}
gene_toId = {}
regG_toID = {}

def import_data(name = "bifurcating_1"):
    global expression, regulators,regulators_network, nsamples, ngenes, genes, id_toGene, gene_toId, regulator_genes,\
        nregulator_genes, regulator_expression, regulator_genes_id

    expression = 0
    regulator_expression = 0
    regulators = 0
    regulators_network = 0
    nsamples = 0
    ngenes = 0
    genes = 0
    regulator_genes = 0
    regulator_genes_id = []
    nregulator_genes = 0
    id_toGene = {}
    gene_toId = {}


    data_name = name + '.csv'
    dir_path = 'C:/Users/pierr/Documents/Université/Master 2/Master Thesis/Dyngen_project/dyngen_manuscript/Datasets/'
    data = pd.read_csv(dir_path + data_name, low_memory=False, sep=',', encoding='utf8')
    expression = data.values
    # regulators = pd.read_csv(dir_path + name + '_regulator.csv', low_memory=False, sep=',', encoding='utf8').values
    regulators_network = pd.read_csv(dir_path + name + '_regulator_network.csv',
                                     low_memory=False, sep=',', encoding='utf8')
    nsamples, ngenes = expression.shape

    genes = pd.read_csv(dir_path + name + '_genes.csv', low_memory=False, sep=',', encoding='utf8').values

    regulator_genes = pd.read_csv(dir_path + name + '_regulator_genes.csv', low_memory=False, sep=',', encoding='utf8').values
    nregulator_genes = regulator_genes.shape[0]

    gene_toId = {}
    id_toGene = {}
    for i in range(0, ngenes):
        gene_toId[genes[i, 0]] = i
        id_toGene[i] = genes[i, 0]


    for i in range(0, nregulator_genes):
        regulator_genes_id.append(gene_toId[regulator_genes[i,0]])
    regulator_expression = expression[:, regulator_genes_id]

    print('done')

def create_dir (name = "", n_estimators = 0, K = 0, min_split = 2):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y %H_%M")
    dir = "Genes_models/Models " + name +' ' + dt_string + "_n_{}".format(n_estimators) + "_K_{}".format(K) +\
          '_msplit_{}'.format(min_split)
    os.mkdir(dir)
    return dir + '/'


# path = Genes_models/
def train_Forests(path = "", n_estimators = 100, K = 20, min_split = 5, name= 'bifurcating_1'):

    import_data(name)
    global nsamples, ngenes, expression, nregulator_genes, regulator_expression, regulator_genes_id
    dir_path = create_dir(name, n_estimators, K, min_split)
    model_name = 'model_{}.joblib'


    # mda_interactions = [[[0.0 for x in range(ngenes)] for y in range(nregulator_genes)] for x in range(nsamples)]
    # mda_interactions = np.array(mda_interactions)

    # mda_interactions = pd.DataFrame(columns=regulators_network.columns)
    mda_interactions = []
    mdi_interactions = []
    SAABAS_interactions = []
    SHAP_interactions = []

    mda_interactionsDF = pd.DataFrame(columns=regulators_network.columns)
    mdi_interactionsDF = pd.DataFrame(columns=regulators_network.columns)
    SHAP_interactionsDF = pd.DataFrame(columns=regulators_network.columns)
    SAABAS_interactionsDF = pd.DataFrame(columns=regulators_network.columns)

    if path == "":
        # Generate models / save them / apply the methods
        for i in range(0, ngenes):
            print("Gene {} in progress".format(i))
            target_gene = id_toGene[i]
            X = regulator_expression
            X = np.float32(X)
            y = expression[:, i]
            # if the regulator gene is part of the expression db, remove it as it now is the target
            if i in regulator_genes_id:
                rm_index = regulator_genes_id.index(i)
                X = np.delete(regulator_expression, rm_index, axis=1)
                X = np.float32(X)

            model = ExtraTreesRegressor(n_estimators=n_estimators, max_features=K, max_depth=None,
                                        min_samples_split=min_split, random_state=0, verbose=0, n_jobs=-1)
            # perfect estimator on training set !!
            model.fit(X, y)
            dump(model, dir_path + 'model_{}'.format(i) + '.joblib')
            # print("Model fitted and dumped")

            # fill here with mda weights
            frequency_control = 'n'
            Uniform = False
            DoSplit = True
            Error_avg = True
            mdaloc = local_mda2.compute_mda_local_ens(model, X, 'classic', frequency_control, uniform=Uniform,
                                                      DoSplit=DoSplit, Error_avg=Error_avg)

            mdaloc = normalize_rankings([mdaloc], norm="l1")[0]

            print("mdi")
            mdiloc = LocalMDI_cy.compute_mdi_local_ens(model, X)
            mdiloc = normalize_rankings([mdiloc], norm="l1")[0]
            print("saabas")
            prediction, bias, contributions = Ti.predict(model, X)
            SAABAS = contributions
            SAABAS = normalize_rankings([SAABAS], norm="l1")[0]
            print(SAABAS)
            print("shap")
            shap_values = shap.TreeExplainer(model).shap_values(X)
            SHAP = shap_values
            SHAP = normalize_rankings([SHAP], norm="l1")[0]


            print("MDA computed")
            best25_cell0_gene2 = np.argsort(mdaloc[0,:])[::-1][0:25]

            iter = regulator_genes_id.copy()
            if i in regulator_genes_id:
                iter.remove(i)
            iter_len = len(iter)

            # print(len(iter))

            for cell in range(0, nsamples):
                for reg in range(0,iter_len):
                    mda_interactions.append(
                        ["cell{}".format(cell+1), id_toGene[iter[reg]], target_gene, mdaloc[cell, reg]])
                    mdi_interactions.append(
                        ["cell{}".format(cell+1), id_toGene[iter[reg]], target_gene, mdiloc[cell, reg]])
                    SAABAS_interactions.append(
                        ["cell{}".format(cell + 1), id_toGene[iter[reg]], target_gene, SAABAS[cell, reg]])
                    SHAP_interactions.append(
                        ["cell{}".format(cell + 1), id_toGene[iter[reg]], target_gene, SHAP[cell, reg]])


            print('DF merged')

            # if i in regulator_genes_id:
            #     for j in range(0, nsamples):
            #         mda_interactions[j, :, i] = np.insert(mdaloc[j, :], obj=i, values=0.0)
            # else:
            #     mda_interactions[:, :, i] = mdaloc

            # mdaloc is of size nsamples x ngenes
            # mdaloc[0,:] give the feature importance ranking of all/0 genes of Cell0 that influence gene 0
            # the ground truth data shows that in general it is 1 to 3 different genes that regulate a gene expression in each cell
            # the ground truth data also shows that many genes are not regulated by at least one gene > 50%
            # some genes are never regulated even among all cells - sum(sum(dt1 != 0.00))
            # np.sort(mda_gene1[0,])[::-1][0:10]
            # regulators_network[regulators_network[:, 0] == 'cell1', :] gives the information of the 150x150 genes
            # interactions matrix for cell1
    else:
        for i in range(1, 2):

            # load model i
            X = np.delete(expression, i, axis=1)
            X = np.float32(X)
            y = expression[:, i]
            model = load(dir_path + model_name.format(i) +'.joblib')
            # fill here with mda weights
            frequency_control = 'n'
            Uniform = False
            DoSplit = True
            Error_avg = True
            mdaloc = local_mda2.compute_mda_local_ens(model, X, 'classic', frequency_control, uniform=Uniform,
                                                      DoSplit=DoSplit, Error_avg=Error_avg)

            best25_cell0_gene2 = np.argsort(mdaloc[0, :])[::-1][0:25]
            # mda_interactions[:, i, :] = mdaloc

    tmpDF = pd.DataFrame(data=mda_interactions, columns=regulators_network.columns)
    mda_interactionsDF = mda_interactionsDF.append(tmpDF)

    tmpDF = pd.DataFrame(data=mdi_interactions, columns=regulators_network.columns)
    mdi_interactionsDF = mdi_interactionsDF.append(tmpDF)

    tmpDF = pd.DataFrame(data=SHAP_interactions, columns=regulators_network.columns)
    SHAP_interactionsDF = SHAP_interactionsDF.append(tmpDF)

    tmpDF = pd.DataFrame(data=SAABAS_interactions, columns=regulators_network.columns)
    SAABAS_interactionsDF = SAABAS_interactionsDF.append(tmpDF)
    print("DF created")
    return mdaloc, model, mda_interactions, mda_interactionsDF, dir_path, mdi_interactionsDF, SAABAS_interactionsDF, SHAP_interactionsDF


def build_comparisonGT(df):

    gt_tmp = df.iloc[:, [0,1,2]].values.tolist()
    gt_interactions = df.iloc[:, [0, 1, 2]]
    values = np.zeros(df.shape[0])

    for index, row in regulators_network.iterrows():
        # row_index = df.index[(df["cell_id"] == row["cell_id"]) & (df["regulator"] == row["regulator"]) &
        #                      (df["target"] == row["target"])].tolist()
        try:
            row_index = gt_tmp.index([row["cell_id"], row["regulator"], row["target"]])
            values[row_index] = row['strength']
        except ValueError:
            print("counld not find a specific element of the ground truth data")
            row_index = -1

        if index % 10000 == 0:
            print('doing')

    gt_interactions.insert(3, regulators_network.columns[3], values, True)
    return gt_interactions


def make_gene_pairs():
    # 150 genes -> 22500 pairs for each cell
    # we consider the 10.000 biggest weights
    # expand GT vector to make comparison
    return


def create_GT_dataset(name = "bifurcating_1"):
    global nsamples, ngenes, genes, regulators_network, gene_toId

    data_regulation = [[[0.0 for x in range(ngenes)] for y in range(ngenes)] for x in range(nsamples)]
    data_regulation = np.array(data_regulation)
    data_interaction = [[[0.0 for x in range(ngenes)] for y in range(ngenes)] for x in range(nsamples)]
    data_interaction = np.array(data_interaction)
    index = np.array(range(0,ngenes))
    dic = gene_toId

    for i in range(0,nsamples):
        for j in regulators_network[regulators_network[:, 0] == 'cell{}'.format(i+1), :]:
            index_regulator = dic[j[1]]
            index_target = dic[j[2]]
            # print(j, index_regulator, index_target, j[3])
            data_regulation[i,index_regulator, index_target] = 1
            data_interaction[i, index_regulator, index_target] = j[3]

    # save data
    data_interaction_reshaped = data_interaction.reshape(data_interaction.shape[0], -1)
    data_regulation_reshaped = data_regulation.reshape(data_regulation.shape[0], -1)
    np.savetxt("Genes_interactions/" + name + "_interaction.gz", data_interaction_reshaped)
    np.savetxt("Genes_interactions/" + name + "_regulation.gz", data_regulation_reshaped)

    return


def load_GT_regulation(name = "bifurcating_1"):
    global ngenes
    data_interaction_reshaped = np.loadtxt("Genes_interactions/" + name + "_interaction.gz")
    data_regulation_reshaped = np.loadtxt("Genes_interactions/" + name + "_regulation.gz")

    data_interaction = data_interaction_reshaped.reshape(
        data_interaction_reshaped.shape[0], data_interaction_reshaped.shape[1] // ngenes, ngenes)

    data_regulation = data_regulation_reshaped.reshape(
        data_regulation_reshaped.shape[0], data_regulation_reshaped.shape[1] // ngenes, ngenes)

    print(data_interaction.shape)
    print(data_regulation.shape)
    return data_interaction, data_regulation

# model_test = load("Genes_models/Models  08_08_2021 00_55_38/model_0")
# X = np.delete(expression, 0, axis=1)
# y = expression[:,0]
# pred = model_test.predict(X)

# return the same dataframe as mdasample with ground truth value in 'strength'
def build_gt_df(mda_df, gtsample):
    gt_tmp = mda_df.iloc[:, [0, 1, 2]].values.tolist()
    gt_interactions = mda_df.iloc[:, [0, 1, 2]]
    values = np.zeros(mda_df.shape[0])
    # print(gtsample.shape)
    for index, row in gtsample.iterrows():
        try:
            row_index = gt_tmp.index([row["cell_id"], row["regulator"], row["target"]])
            # values[row_index] = row['strength']
            values[row_index] = 1.0
        except ValueError:
            # print("counld not find a specific element of the ground truth data")
            # print(row["cell_id"], row["regulator"], row["target"])
            row_index = -1
    gt_interactions.insert(3, 'strength', values)
    return gt_interactions


from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score
def compute_scores(mdadf):

    global nsamples, regulators_network
    nsamples2 = 10
    aupr = np.zeros(nsamples2)
    avgpr = np.zeros(nsamples2)
    auroc = np.zeros(nsamples2)

    for i in range(0,nsamples2):
        # compute metrics for each cell individually
        # report the mean
        mda_sample = mdadf.loc[mdadf['cell_id'] == "cell{}".format(i+1)]
        gt_sample = regulators_network.loc[regulators_network['cell_id'] == "cell{}".format(i+1)]
        gt_sample = build_gt_df(mda_sample, gt_sample)
        # print(gt_sample)
        # print(mda_sample)
        gt_sample = gt_sample['strength'].values.tolist()
        mda_sample = mda_sample['strength'].values.tolist()

        for index, item in enumerate(mda_sample):
            if item > 1.0:
                mda_sample[index] = 1.0

        if i%10 == 0:
            print("Doing performance measurements {} %".format(i/1000))

        # gt_sample[np.nonzero(gt_sample)] = 1.0
        precision, recall, thresholds = precision_recall_curve(gt_sample, mda_sample, pos_label=1)

        avgpr[i] = average_precision_score(gt_sample, mda_sample)
        precision, recall = zip(*sorted(zip(precision, recall)))
        aupr[i] = auc(precision, recall)
        auroc[i] = roc_auc_score(gt_sample, mda_sample)

    print("mean AUPR (auc method): {}".format(np.round(np.mean(aupr), 3)))
    print("mean AUPR: {}".format(np.round(np.mean(avgpr), 3)))
    print("mean AUROC: {}".format(np.round(np.mean(auroc), 3)))

    return aupr, auroc

def compute_global_scores():
    # datasets = ['bifurcating_1', "bifurcating_loop_1", "bifurcating_converging_1", "bifurcating_cycle_1",
    #             "binary_tree_1", "consecutive_bifurcating_1", "cycle_1",
    #             "cycle_simple_1", "converging_1"]
    datasets = ["converging_1"]
    nbr = len(datasets)
    import_data("converging_1")
    file_path = 'Genes_models\local_measures' + '/'

    for i in range(0,nbr):
        mda_data = pd.read_csv(file_path + datasets[i] + "_MDA_interactions.csv", low_memory=False, sep=',', encoding='utf8')
        mdi_data = pd.read_csv(file_path + datasets[i] + "_MDI_interactions.csv", low_memory=False, sep=',', encoding='utf8')
        shap_data = pd.read_csv(file_path + datasets[i] + "_SHAP_interactions.csv", low_memory=False, sep=',', encoding='utf8')
        saabas_data = pd.read_csv(file_path + datasets[i] + "_SAABAS_interactions.csv", low_memory=False, sep=',', encoding='utf8')
        # the score is compute for each cell, then average are repored in auroc and aupr vectors
        nsamples = 1000
        # aupr_SSN = np.zeros(nsamples)
        # aupr_pySCENIC = np.zeros(nsamples)
        # aupr_Lioness = np.zeros(nsamples)
        # auroc_SSN = np.zeros(nsamples)
        # auroc_pySCENIC = np.zeros(nsamples)
        # auroc_Lioness = np.zeros(nsamples)
        aupr_mda = np.zeros(nsamples)
        auroc_mda = np.zeros(nsamples)
        aupr_mdi = np.zeros(nsamples)
        auroc_mdi = np.zeros(nsamples)
        aupr_shap = np.zeros(nsamples)
        auroc_shap = np.zeros(nsamples)
        aupr_saabas = np.zeros(nsamples)
        auroc_saabas = np.zeros(nsamples)
        print('Start cell evaluation')
        dir_path = 'C:/Users/pierr/Documents/Université/Master 2/Master Thesis/Dyngen_project/dyngen_manuscript/Datasets/'
        regulators_network = pd.read_csv(dir_path + datasets[i] + '_regulator_network.csv',
                                         low_memory=False, sep=',', encoding='utf8')

        dictionnary = draw_unique_dictionnary()
        print(datasets[i])

        mda_sample = mda_data['strength'].values
        mdi_sample = [abs(ele) for ele in mdi_data['strength'].values]
        shap_sample = [abs(ele1) for ele1 in shap_data['strength'].values]
        saabas_sample = [abs(ele2) for ele2 in saabas_data['strength'].values]

        mda_sample = make_global(mda_sample)
        mdi_sample = make_global(mdi_sample)
        shap_sample = make_global(shap_sample)
        saabas_sample = make_global(saabas_sample)

        for j in range(0, nsamples):
            # locate the ranking for each cell
            gt_sample = regulators_network.loc[regulators_network['cell_id'] == "cell{}".format(j + 1)]
            gt_sample = ground_truth_vector(gt_sample, mda_sample, dictionnary)

            if j % 100 == 0:
                print("Doing performance measurements {} %".format(j / 1000))

            auroc_mda[j] = roc_auc_score(gt_sample, mda_sample)
            auroc_mdi[j] = roc_auc_score(gt_sample, mdi_sample)
            auroc_saabas[j] = roc_auc_score(gt_sample, saabas_sample)
            auroc_shap[j] = roc_auc_score(gt_sample, shap_sample)

            precision, recall, thresholds = precision_recall_curve(gt_sample, mda_sample, pos_label=1)
            precision, recall = zip(*sorted(zip(precision, recall)))
            aupr_mda[j] = auc(precision, recall)

            precision, recall, thresholds = precision_recall_curve(gt_sample, mdi_sample, pos_label=1)
            precision, recall = zip(*sorted(zip(precision, recall)))
            aupr_mdi[j] = auc(precision, recall)

            precision, recall, thresholds = precision_recall_curve(gt_sample, shap_sample, pos_label=1)
            precision, recall = zip(*sorted(zip(precision, recall)))
            aupr_shap[j] = auc(precision, recall)

            precision, recall, thresholds = precision_recall_curve(gt_sample, saabas_sample, pos_label=1)
            precision, recall = zip(*sorted(zip(precision, recall)))
            aupr_saabas[j] = auc(precision, recall)

        print("mean AUROC MDA: {}".format(np.round(np.mean(auroc_mda), 4)))
        print("mean AUROC MDI: {}".format(np.round(np.mean(auroc_mdi), 4)))
        print("mean AUROC SHAP: {}".format(np.round(np.mean(auroc_shap), 4)))
        print("mean AUROC SAABAS: {}".format(np.round(np.mean(auroc_saabas), 4)))
        print("std AUROC MDA: {}".format(np.round(np.std(auroc_mda), 4)))
        print("std AUROC MDI: {}".format(np.round(np.std(auroc_mdi), 4)))
        print("std AUROC SHAP: {}".format(np.round(np.std(auroc_shap), 4)))
        print("std AUROC SAABAS: {}".format(np.round(np.std(auroc_saabas), 4)))

        print("mean AUPR MDA: {}".format(np.round(np.mean(aupr_mda), 4)))
        print("mean AUPR MDI: {}".format(np.round(np.mean(aupr_mdi), 4)))
        print("mean AUPR SHAP: {}".format(np.round(np.mean(aupr_shap), 4)))
        print("mean AUPR SAABAS: {}".format(np.round(np.mean(aupr_saabas), 4)))
        print("std AUPR MDA: {}".format(np.round(np.std(aupr_mda), 4)))
        print("std AUPR MDI: {}".format(np.round(np.std(aupr_mdi), 4)))
        print("std AUPR SHAP: {}".format(np.round(np.std(aupr_shap), 4)))
        print("std AUPR SAABAS: {}".format(np.round(np.std(aupr_saabas), 4)))
    return


def display_paper_method_results(names=['bifurcating_1', "bifurcating_loop_1", "bifurcating_converging_1", "bifurcating_cycle_1"]):

    nbr = len(names)
    print(nbr)
    auroc_ssn = np.zeros(nbr)
    aupr_ssn = np.zeros(nbr)
    auroc_pyscenic = np.zeros(nbr)
    aupr_pyscenic = np.zeros(nbr)
    auroc_lion = np.zeros(nbr)
    aupr_lion = np.zeros(nbr)
    F1_ssn = np.zeros(nbr)
    F1_pyscenic = np.zeros(nbr)
    F1_lion = np.zeros(nbr)

    dir_path = 'C:/Users/pierr/Documents/Université/Master 2/Master Thesis/Dyngen_project/dyngen_manuscript/Datasets/'
    data = pd.read_csv(dir_path + 'bifurcating_1_scores' + '.csv', low_memory=False, sep=',', encoding='utf8')
    for i in range(0, nbr):
        # data = pd.read_csv(dir_path + 'bifurcating_1_scores' + '.csv', low_memory=False, sep=',', encoding='utf8')
        scores = data.loc[data['dataset_id'].values == names[i], ]
        auroc_ssn[i], aupr_ssn[i], F1_ssn[i] = np.mean(scores.loc[scores['cni_method_id'].values == 'ssn'])
        auroc_pyscenic[i], aupr_pyscenic[i], F1_pyscenic[i] = np.mean(scores.loc[scores['cni_method_id'].values == 'pyscenic'])
        auroc_lion[i], aupr_lion[i], F1_lion[i] = np.mean(scores.loc[scores['cni_method_id'].values == 'lionesspearson'])

    fig, axs = plt.subplots(1, 3)
    axs[0].plot(auroc_ssn, aupr_ssn, '.', color='blue')
    axs[0].set_title('SSN')
    axs[1].plot(auroc_pyscenic, aupr_pyscenic, '.', color='red')
    axs[1].set_title('pyScenic')
    axs[2].plot(auroc_lion, aupr_lion, '.', color='green')
    axs[2].set_title('Lionness + Pearson')

    for ax in axs.flat:
        ax.set(xlabel='meanAUROC', ylabel='meanAUPR')

    return auroc_ssn.ravel(), aupr_ssn, auroc_pyscenic, aupr_pyscenic, auroc_lion,aupr_lion


def analysis_draw_measures():

    print("start MDA")
    # datasets = ['bifurcating_1', "bifurcating_loop_1", "bifurcating_converging_1", "bifurcating_cycle_1",
    #             "binary_tree_1", "consecutive_bifurcating_1", "cycle_1",
    #             "cycle_simple_1", "converging_1"]
    datasets = ["converging_1"]
    nbr = len(datasets)

    for i in range(0, nbr):
        print("Computing dataset: {}".format(i))

        import_data(datasets[i])
        K = len(regulator_genes) -4
        _, _, _, inter_df, dir_path, inter_dfMDI, inter_dfSAABAS, inter_dfSHAP = train_Forests(
            n_estimators=40, K=K, min_split=5, name=datasets[i])
        inter_df.to_csv(path_or_buf=dir_path + datasets[i] + "_MDA_interactions.csv", index=False)
        inter_dfMDI.to_csv(path_or_buf=dir_path + datasets[i] + "_MDI_interactions.csv", index=False)
        inter_dfSAABAS.to_csv(path_or_buf=dir_path + datasets[i] + "_SAABAS_interactions.csv", index=False)
        inter_dfSHAP.to_csv(path_or_buf=dir_path + datasets[i] + "_SHAP_interactions.csv", index=False)

    return


def method_comparison_script():
    # datasets = ['bifurcating_1', "bifurcating_loop_1", "bifurcating_converging_1", "bifurcating_cycle_1",
    #             "binary_tree_1", "consecutive_bifurcating_1", "cycle_1",
    #             "cycle_simple_1", "converging_1"]
    datasets = ["converging_1"]
    nbr = len(datasets)
    import_data("converging_1")

    file_path = 'Genes_models\local_measures' + '/'

    # mean_aupr_SSN = np.zeros(nbr)
    # mean_aupr_pySCENIC = np.zeros(nbr)
    # mean_aupr_Lioness = np.zeros(nbr)
    # mean_auroc_SSN = np.zeros(nbr)
    # mean_auroc_pySCENIC = np.zeros(nbr)
    # mean_auroc_Lioness = np.zeros(nbr)
    # mean_aupr_mda = np.zeros(nbr)
    # mean_auroc_mda = np.zeros(nbr)
    # mean_aupr_mdi = np.zeros(nbr)
    # mean_auroc_mdi = np.zeros(nbr)
    # mean_aupr_shap = np.zeros(nbr)
    # mean_auroc_shap = np.zeros(nbr)
    # mean_aupr_saabas = np.zeros(nbr)
    # mean_auroc_saabas = np.zeros(nbr)

    for i in range(0,nbr):
        mda_data = pd.read_csv(file_path + datasets[i] + "_MDA_interactions.csv", low_memory=False, sep=',', encoding='utf8')
        mdi_data = pd.read_csv(file_path + datasets[i] + "_MDI_interactions.csv", low_memory=False, sep=',', encoding='utf8')
        shap_data = pd.read_csv(file_path + datasets[i] + "_SHAP_interactions.csv", low_memory=False, sep=',', encoding='utf8')
        saabas_data = pd.read_csv(file_path + datasets[i] + "_SAABAS_interactions.csv", low_memory=False, sep=',', encoding='utf8')
        # the score is compute for each cell, then average are repored in auroc and aupr vectors
        nsamples = 1000
        # aupr_SSN = np.zeros(nsamples)
        # aupr_pySCENIC = np.zeros(nsamples)
        # aupr_Lioness = np.zeros(nsamples)
        # auroc_SSN = np.zeros(nsamples)
        # auroc_pySCENIC = np.zeros(nsamples)
        # auroc_Lioness = np.zeros(nsamples)
        aupr_mda = np.zeros(nsamples)
        auroc_mda = np.zeros(nsamples)
        aupr_mdi = np.zeros(nsamples)
        auroc_mdi = np.zeros(nsamples)
        aupr_shap = np.zeros(nsamples)
        auroc_shap = np.zeros(nsamples)
        aupr_saabas = np.zeros(nsamples)
        auroc_saabas = np.zeros(nsamples)
        print('Start cell evaluation')

        dir_path = 'C:/Users/pierr/Documents/Université/Master 2/Master Thesis/Dyngen_project/dyngen_manuscript/Datasets/'
        regulators_network = pd.read_csv(dir_path + datasets[i] + '_regulator_network.csv',
                                         low_memory=False, sep=',', encoding='utf8')

        dictionnary = draw_unique_dictionnary()
        print(datasets[i])
        for j in range(0, nsamples):
            # locate the ranking for each cell
            mda_sample = mda_data.loc[mda_data['cell_id'] == "cell{}".format(j + 1)]
            mdi_sample = mdi_data.loc[mdi_data['cell_id'] == "cell{}".format(j + 1)]
            shap_sample = shap_data.loc[shap_data['cell_id'] == "cell{}".format(j + 1)]
            saabas_sample = saabas_data.loc[saabas_data['cell_id'] == "cell{}".format(j + 1)]
            # locate the gt data for each cell

            gt_sample = regulators_network.loc[regulators_network['cell_id'] == "cell{}".format(j + 1)]
            gt_sample = ground_truth_vector(gt_sample, mda_sample, dictionnary)

            mda_sample = mda_sample['strength'].values.tolist()
            mdi_sample = [abs(ele) for ele in mdi_sample['strength'].values.tolist()]
            shap_sample = [abs(ele1) for ele1 in shap_sample['strength'].values.tolist()]
            saabas_sample = [abs(ele2) for ele2 in saabas_sample['strength'].values.tolist()]

            if j % 100 == 0:
                print("Doing performance measurements {} %".format(j / 1000))

            auroc_mda[j] = roc_auc_score(gt_sample, mda_sample)
            auroc_mdi[j] = roc_auc_score(gt_sample, mdi_sample)
            auroc_saabas[j] = roc_auc_score(gt_sample, saabas_sample)
            auroc_shap[j] = roc_auc_score(gt_sample, shap_sample)

            precision, recall, thresholds = precision_recall_curve(gt_sample, mda_sample, pos_label=1)
            precision, recall = zip(*sorted(zip(precision, recall)))
            aupr_mda[j] = auc(precision, recall)

            precision, recall, thresholds = precision_recall_curve(gt_sample, mdi_sample, pos_label=1)
            precision, recall = zip(*sorted(zip(precision, recall)))
            aupr_mdi[j] = auc(precision, recall)

            precision, recall, thresholds = precision_recall_curve(gt_sample, shap_sample, pos_label=1)
            precision, recall = zip(*sorted(zip(precision, recall)))
            aupr_shap[j] = auc(precision, recall)

            precision, recall, thresholds = precision_recall_curve(gt_sample, saabas_sample, pos_label=1)
            precision, recall = zip(*sorted(zip(precision, recall)))
            aupr_saabas[j] = auc(precision, recall)

        print("mean AUROC MDA: {}".format(np.round(np.mean(auroc_mda), 4)))
        print("mean AUROC MDI: {}".format(np.round(np.mean(auroc_mdi), 4)))
        print("mean AUROC SHAP: {}".format(np.round(np.mean(auroc_shap), 4)))
        print("mean AUROC SAABAS: {}".format(np.round(np.mean(auroc_saabas), 4)))
        print("std AUROC MDA: {}".format(np.round(np.std(auroc_mda), 4)))
        print("std AUROC MDI: {}".format(np.round(np.std(auroc_mdi), 4)))
        print("std AUROC SHAP: {}".format(np.round(np.std(auroc_shap), 4)))
        print("std AUROC SAABAS: {}".format(np.round(np.std(auroc_saabas), 4)))

        print("mean AUPR MDA: {}".format(np.round(np.mean(aupr_mda), 4)))
        print("mean AUPR MDI: {}".format(np.round(np.mean(aupr_mdi), 4)))
        print("mean AUPR SHAP: {}".format(np.round(np.mean(aupr_shap), 4)))
        print("mean AUPR SAABAS: {}".format(np.round(np.mean(aupr_saabas), 4)))
        print("std AUPR MDA: {}".format(np.round(np.std(aupr_mda), 4)))
        print("std AUPR MDI: {}".format(np.round(np.std(aupr_mdi), 4)))
        print("std AUPR SHAP: {}".format(np.round(np.std(aupr_shap), 4)))
        print("std AUPR SAABAS: {}".format(np.round(np.std(aupr_saabas), 4)))


        # print("median AUROC MDA: {}".format(np.round(np.median(auroc_mda), 3)))
        # print("median AUROC MDI: {}".format(np.round(np.median(auroc_mdi), 3)))
        # print("median AUROC SHAP: {}".format(np.round(np.median(auroc_shap), 3)))
        # print("median AUROC SAABAS: {}".format(np.round(np.median(auroc_saabas), 3)))
        #
        # mean_auroc_mda[i] = np.round(np.mean(auroc_mda), 3)
        # mean_auroc_mdi[i] = np.round(np.mean(auroc_mdi), 3)
        # mean_auroc_shap[i] = np.round(np.mean(auroc_shap), 3)
        # mean_auroc_saabas[i] = np.round(np.mean(auroc_saabas), 3)
    return

def draw_AUPR_AUROC_graphs():
    nbr = 5

    datasets = ['bifurcating_1', "bifurcating_loop_1", "bifurcating_converging_1", "bifurcating_cycle_1",
                "binary_tree_1", "consecutive_bifurcating_1", "cycle_1",
                "cycle_simple_1", "converging_1"]
    mean_auroc_SSN, mean_aupr_SSN, mean_auroc_pySCENIC, mean_aupr_pySCENIC, mean_auroc_Lioness, mean_aupr_Lioness = display_paper_method_results(datasets)

    mean_aupr_mda = [0.1012, 0.0773,0.0945, 0.0823, 0.0656, 0.0762, 0.0793, 0.166, 0.1593]
    mean_auroc_mda = [0.8498, 0.6451, 0.6868, 0.6516, 0.901, 0.912, 0.717, 0.7834, 0.6934]

    mean_aupr_mdi = [0.1099, 0.0552, 0.0743, 0.0861, 0.2053, 0.2189, 0.1231, 0.1885, 0.0981]
    mean_auroc_mdi = [ 0.8755,  0.625,0.714, 0.6759, 0.9215, 0.9345, 0.7803, 0.8019, 0.6934]

    mean_aupr_shap = [0.1392, 0.0423,0.0384, 0.0333, 0.1079, 0.1376, 0.1219, 0.1174, 0.0561]
    mean_auroc_shap = [0.8475, 0.6351,0.6864, 0.6423, 0.9269,0.9352, 0.7667, 0.811, 0.6493]

    mean_aupr_saabas = [0.1242, 0.0405,0.0415, 0.0313, 0.144, 0.1443, 0.0804, 0.1246, 0.0528]
    mean_auroc_saabas = [0.8519, 0.6283, 0.6842, 0.6448, 0.9155, 0.9229, 0.7572, 0.8042, 0.6433]

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    axs[0].plot(mean_auroc_SSN, mean_aupr_SSN, '.', color='blue')
    axs[0].plot(mean_auroc_pySCENIC, mean_aupr_pySCENIC, '.', color='grey')
    axs[0].plot(mean_auroc_Lioness, mean_aupr_Lioness, '.', color='grey')
    axs[0].set_title('SSN')
    axs[1].plot(mean_auroc_pySCENIC, mean_aupr_pySCENIC, '.', color='red')
    axs[1].plot(mean_auroc_Lioness, mean_aupr_Lioness, '.', color='grey')
    axs[1].plot(mean_auroc_SSN, mean_aupr_SSN, '.', color='grey')
    axs[1].set_title('pyScenic')
    axs[2].plot(mean_auroc_Lioness, mean_aupr_Lioness, '.', color='green')
    axs[2].plot(mean_auroc_pySCENIC, mean_aupr_pySCENIC, '.', color='grey')
    axs[1].plot(mean_auroc_SSN, mean_aupr_SSN, '.', color='grey')
    axs[2].set_title('LIONESS + Pearson')
    for ax in axs.flat:
        ax.set(xlabel='meanAUROC', ylabel='meanAUPR')
    plt.xlim([0.4, 1])
    plt.ylim([0, 0.1])

    fig2, axs2 = plt.subplots(1, 4, sharex=True, sharey=True)
    axs2[0].plot(mean_auroc_mda, mean_aupr_mda, '.', color='blue')
    axs2[0].plot(mean_auroc_mdi, mean_aupr_mdi, '.', color='grey')
    axs2[0].plot(mean_auroc_saabas, mean_aupr_saabas, '.', color='grey')
    axs2[0].plot(mean_auroc_shap, mean_aupr_shap, '.', color='grey')
    axs2[0].plot(mean_auroc_pySCENIC, mean_aupr_pySCENIC, '.', color='cyan')
    axs2[0].set_title('local MDA')

    axs2[1].plot(mean_auroc_mdi, mean_aupr_mdi, '.', color='green')
    axs2[1].plot(mean_auroc_mda, mean_aupr_mda, '.', color='grey')
    axs2[1].plot(mean_auroc_saabas, mean_aupr_saabas, '.', color='grey')
    axs2[1].plot(mean_auroc_shap, mean_aupr_shap, '.', color='grey')
    axs2[1].plot(mean_auroc_pySCENIC, mean_aupr_pySCENIC, '.', color='cyan')
    axs2[1].set_title('local MDI')

    axs2[2].plot(mean_auroc_shap, mean_aupr_shap, '.', color='red')
    axs2[2].plot(mean_auroc_mda, mean_aupr_mda, '.', color='grey')
    axs2[2].plot(mean_auroc_mdi, mean_aupr_mdi, '.', color='grey')
    axs2[2].plot(mean_auroc_saabas, mean_aupr_saabas, '.', color='grey')
    axs2[2].plot(mean_auroc_pySCENIC, mean_aupr_pySCENIC, '.', color='cyan')
    axs2[2].set_title('SHAP')

    axs2[3].plot(mean_auroc_saabas, mean_aupr_saabas, '.', color='orange')
    axs2[3].plot(mean_auroc_mda, mean_aupr_mda, '.', color='grey')
    axs2[3].plot(mean_auroc_mdi, mean_aupr_mdi, '.', color='grey')
    axs2[3].plot(mean_auroc_shap, mean_aupr_shap, '.', color='grey')
    axs2[3].plot(mean_auroc_pySCENIC, mean_aupr_pySCENIC, '.', color='cyan')
    axs2[3].set_title('SAABAS')
    for ax in axs2.flat:
        ax.set(xlabel='meanAUROC', ylabel='meanAUPR')
    plt.xlim([0.4, 1])
    plt.ylim([0, 0.21])


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    # Creating axes instance
    data = [mean_auroc_mda, mean_auroc_mdi, mean_auroc_shap, mean_auroc_saabas, mean_auroc_pySCENIC]
    bp = ax.boxplot(data, patch_artist=True, vert=0)
    colors = ['blue', 'green',
              'red', 'orange', 'cyan']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    # x-axis labels
    ax.set_yticklabels(['MDA', 'MDI',
                        'SHAP', 'SAABAS', 'pySCENIC'])
    # Adding title
    plt.title("meanAUROC scores Boxplot")
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # show plot
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    # Creating axes instance
    data = [mean_aupr_mda, mean_aupr_mdi, mean_aupr_shap, mean_aupr_saabas, mean_aupr_pySCENIC]
    bp = ax.boxplot(data, patch_artist=True, vert=0)
    colors = ['blue', 'green',
              'red', 'orange', 'cyan']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    # x-axis labels
    ax.set_yticklabels(['MDA', 'MDI',
                        'SHAP', 'SAABAS', 'pySCENIC'])
    # Adding title
    plt.title("meanAUPR scores Boxplot")
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # show plot
    plt.show()


def make_global(method):

    size = method.shape[0] / 1000
    values = np.zeros(int(size))
    for j in range(0, 1000):
        data = method.loc[method['cell_id'] == "cell{}".format(j + 1)]
        data = [abs(ele) for ele in data['strength'].values.tolist()]
        for i in range(0, int(size)):
            values[i] = values[i] + data[i]
    return values/1000


def ground_truth_vector(gt_sample, mda_df, dict):
    gt_interactions = mda_df.iloc[:, [0, 1, 2]]
    gt_tmp = mda_df.iloc[:, [0, 1, 2]].values.tolist()
    values = np.zeros(mda_df.shape[0])
    valuesBIS = np.zeros(mda_df.shape[0])
    tmp = 0
    # for index, row in gt_sample.iterrows():
    for i in range(0, gt_sample.shape[0]):
        reg = gt_sample.iloc[i]['regulator']
        target = gt_sample.iloc[i]['target']
        if reg != target:
            # test_index = get_row_index(row["regulator"], row["target"])
            test_index = dict[reg, target]
            valuesBIS[test_index] = 1.0

        # if row["regulator"] != row["target"]:
        #     # test_index = get_row_index(row["regulator"], row["target"])
        #     test_index = dict[row["regulator"], row["target"]]
        #     valuesBIS[test_index] = 1.0

    # print(tmp)
    # print(gt_sample.shape)
    return valuesBIS

def draw_unique_dictionnary():
    dict = {}
    tmp_data = regulators_network.iloc[:, [1, 2]].drop_duplicates(inplace=False)
    for index, row in tmp_data.iterrows():
        dict[row["regulator"], row["target"]] = get_row_index(row["regulator"], row["target"])
    print(dict)
    return dict


def get_row_index(regulator, target):

    target_index = gene_toId[target]
    index1 = target_index * nregulator_genes
    tmp1 = 0
    for i in range(0,target_index):
        if i in regulator_genes_id:
            tmp1 = tmp1 +1
    index1 = index1 - tmp1

    # now find the position through of the gene regulator
    regulator_id = gene_toId[regulator]
    regulator_position = regulator_genes_id.index(regulator_id)
    tmp2 = 0
    if target_index in regulator_genes_id[0:regulator_position]:
        tmp2 = 1
    regulator_position = regulator_position - tmp2
    return index1 + regulator_position

def analysis():
    print("start MDA")
    datasets = ['bifurcating_1', "bifurcating_loop_1", "bifurcating_converging_1", "bifurcating_cycle_1"]
    df_set_40_20 = []
    nbr = len(datasets)
    aupr = np.zeros(nbr)
    auroc = np.zeros(nbr)

    # for i in range(0, nbr):
    #     print("Computing dataset: {}".format(i))
    #
    #     import_data(datasets[i])
    #     K = len(regulator_genes) -4
    #     _, _, _, inter_df, dir_path, inter_dfMDI, inter_dfSAABAS, inter_dfSHAP = train_Forests(
    #         n_estimators=40, K = K, min_split = 5, name=datasets[i])
    #     inter_df.to_csv(path_or_buf=dir_path + datasets[i] + "_MDA_interactions.csv", index=False)
    #     inter_dfMDI.to_csv(path_or_buf=dir_path + datasets[i] + "_MDI_interactions.csv", index=False)
    #     inter_dfSAABAS.to_csv(path_or_buf=dir_path + datasets[i] + "_SAABAS_interactions.csv", index=False)
    #     inter_dfSHAP.to_csv(path_or_buf=dir_path + datasets[i] + "_SHAP_interactions.csv", index=False)
    #     aupr[i], auroc[i] = compute_scores(inter_df)

    datasets = ['bifurcating_1', "bifurcating_loop_1", "bifurcating_converging_1", "bifurcating_cycle_1",
                "binary_tree_1", "consecutive_bifurcating_1", "cycle_1",
                "cycle_simple_1", "converging_1"]
    aupr = [0.06,0.02 , 0.016 ,0]
    auroc = [0.8, 0.623 , 0.635 ,0]
    auroc_ssn, aupr_ssn, auroc_pyscenic, aupr_pyscenic, auroc_lion, aupr_lion = display_paper_method_results(datasets)
    fig, axs = plt.subplots(1, 4)
    axs[0].plot(auroc_ssn, aupr_ssn, '.', color='blue')
    axs[0].set_title('SSN')
    axs[1].plot(auroc_pyscenic, aupr_pyscenic, '.', color='red')
    axs[1].set_title('pyScenic')
    axs[2].plot(auroc_lion, aupr_lion, '.', color='green')
    axs[2].set_title('Lionness + Pearson')

    axs[3].plot(auroc, aupr, '.', color='blue')
    axs[3].set_title('local_mda')

    print('auroc ssn:', auroc_ssn)
    print('auroc liones:', auroc_lion)
    print('auroc scenic:', auroc_pyscenic)
    print('aupr ssn:', aupr_ssn)
    print('aupr liones:', aupr_lion)
    print('aupr scenic:', aupr_pyscenic)
    for ax in axs.flat:
        ax.set(xlabel='meanAUROC', ylabel='meanAUPR')



    return aupr, auroc

if __name__ == "__main__":
    import_data("bifurcating_1")
    # datasets = ["bifurcating_loop_1", "bifurcating_converging_1", "bifurcating_cycle_1"]
    # i = 2
    # import_data(datasets[i])
    # print("start MDA")
    # mda_gene, model, inter, interDF, dir_path = train_Forests(n_estimators=40, K=60, min_split=5, name=datasets[i])
    # interDF.to_csv(path_or_buf=dir_path + "MDI_interactions.csv", index=False)
    # # interDF = pd.read_csv(dir_path + "MDA_interactions.csv", low_memory=False, sep=',', encoding='utf8')
    # # interDF = pd.read_csv('Genes_models/Models  09_08_2021 15_11_n_50_K_40_msplit_5/' + "MDA_interactions.csv", low_memory=False, sep=',', encoding='utf8')
    # # interDF.to_excel(dir_path + "MDA_interactions.xlsx", index=False)
    #
    #
    # KKmda_gene, KKmodel, KKinter, KKinterDF, KKdir_path = train_Forests(n_estimators=500, K = 35, min_split = 5)
    # KKinterDF.to_csv(path_or_buf=dir_path + "MDA_interactions.csv", index=False)


    print("Build GT")
    # gtdd = build_comparisonGT(interDF)
    # gtdd.to_csv(path_or_buf=dir_path + "GT_interactions.csv", index=False)
    # gtdd.to_excel(path_or_buf=dir_path + "GT_interactions.csv", index=False)
