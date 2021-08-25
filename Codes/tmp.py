import numpy as np
#
# Official Code for Classification_local_mda
#
counter = 0
counter1 = 0
counter2 = 0

def prob_node_left(tree, node, uniform):
    # [tree.children_left[node]] = node + 1 as long as node is not a leaf
    if uniform:
        return 0.50
    else:
        return tree.n_node_samples[tree.children_left[node]] / tree.n_node_samples[node]

def exploreC(tree, node, children_right_view, sample, feature_view, x, uniform):

    global counter, counter1, counter2
    # Stop exploration as we meet a leaf
    if tree.feature[node] == -2:
        value = tree.value[node].ravel() / tree.value[node].max()
        return value
    # else check current split and define next direction
    # Propagate the sample in both directions if the feature of interest is tested at #node = 'node'

    elif feature_view[node] == x:
        counter += 1
        l_prob = prob_node_left(tree, node, uniform)
        r_prob = 1 - l_prob
        value1 = exploreC(tree, node + 1, children_right_view, sample, feature_view, x, uniform)
        value2 = exploreC(tree, children_right_view[node], children_right_view, sample, feature_view, x, uniform)
        return l_prob * value1 + r_prob * value2
    else:
        if sample[tree.feature[node]] <= tree.threshold[node]:
            counter1 += 1
            # left child is next node
            return exploreC(tree, node+1, children_right_view, sample, feature_view, x, uniform)
        else:
            counter2 += 1
            return exploreC(tree, children_right_view[node], children_right_view, sample, feature_view, x, uniform)


def weighted_prediction_C(tree, prediction, nodes_id, leaf, children_right_view, sample, feature_view, x, uniform):
    dir = 'left'
    node = nodes_id[0]
    # direction of sample at split
    if leaf >= children_right_view[node]:
        dir = 'right'

    l_prob = prob_node_left(tree, nodes_id[0], uniform)
    r_prob = 1 - l_prob

    if l_prob > 1:
        print("Error lprob")
    if r_prob > 1:
        print("Error rightprob")

    # print(l_prob, "  ", r_prob)
    # print(dir)
    if nodes_id.size == 1:
        if dir == 'left':
            # propagate the sample to the right as the decision path goes left at #node = 'node'
            value = (prediction * l_prob) + (r_prob * exploreC(tree, children_right_view[node], children_right_view, sample, feature_view, x, uniform))
            return (prediction * l_prob) + (r_prob * exploreC(tree, children_right_view[node], children_right_view, sample, feature_view, x, uniform))
        else:
            # propagate the sample to the left as the decision path goes right
            value = (prediction * r_prob) + (l_prob * exploreC(tree, node + 1, children_right_view, sample, feature_view, x, uniform))
            return (prediction * r_prob) + (l_prob * exploreC(tree, node + 1, children_right_view, sample, feature_view, x, uniform))
    nodes_id = nodes_id[1:]
    if dir == 'left':
        value1 = r_prob * exploreC(tree, children_right_view[node], children_right_view, sample, feature_view, x, uniform)
        value2 = l_prob * weighted_prediction_C(tree, prediction, nodes_id, leaf, children_right_view, sample, feature_view, x, uniform)
        return (r_prob * exploreC(tree, children_right_view[node], children_right_view, sample, feature_view, x, uniform)) + (l_prob * weighted_prediction_C(tree, prediction, nodes_id, leaf, children_right_view, sample, feature_view, x, uniform))
    else:
        value = (l_prob * exploreC(tree, node + 1, children_right_view, sample, feature_view, x, uniform)) + (r_prob * weighted_prediction_C(tree, prediction, nodes_id, leaf, children_right_view, sample, feature_view, x, uniform))
        return (l_prob * exploreC(tree, node + 1, children_right_view, sample, feature_view, x, uniform)) + (r_prob * weighted_prediction_C(tree, prediction, nodes_id, leaf, children_right_view, sample, feature_view, x, uniform))


def compute_mda_local_treeC(Ctree, X, nsamples, nfeatures,nclass, vimp, uniform):
    # use Ctree.value[node]
    children_left_view = Ctree.children_left
    children_right_view = Ctree.children_right
    feature_view = Ctree.feature
    feature_range = list(range(nfeatures))
    node_indicator = Ctree.decision_path(X)
    node = 0
    ifeat = 0

    for i in range(nsamples):
        # features of decision path (from 0 to n_features-1)
        # Discard leaf from node path
        features = feature_view[node_indicator[i, :].indices][:-1]
        unique_f = np.unique(features)
        # node id of decision path
        nodes = node_indicator[i, :].indices[:-1]
        prediction = Ctree.predict(X)[i].ravel()
        prediction = prediction/ np.max(prediction)
        leaf = node_indicator[i, :].indices[-1]
        for x in unique_f:
            # nodes id of the decision path of sample i that test feature 'x'
            nodes_id = nodes[features == x]
            val = weighted_prediction_C(Ctree, prediction, nodes_id, leaf, children_right_view, X[i], feature_view, x, uniform)
            vimp[i, x, :] = vimp[i, x, :] + val
    return vimp


def compute_mda_local_ens_C(ens, X, uniform):
    nsamples = X.shape[0]
    nfeatures = X.shape[1]
    nclass = ens.classes_.size
    vimp = [[[0.0 for x in range(nclass)] for y in range(nfeatures)] for x in range(nsamples)]
    vimp = np.array(vimp)
    # Dim1 : samples, Dim2: feature, Dim3: class
    nestimators = ens.n_estimators

    for i in range(nestimators):
        # print("o", end='', flush=True)
        vimp = compute_mda_local_treeC(ens.estimators_[i].tree_, X, nsamples, nfeatures,nclass, vimp, uniform)

    print("")
    vimp /= (ens.n_estimators)

    for i in range(0,nsamples):
        for j in range(0,nfeatures):
            vimp[i, j, :] = vimp[i, j, :] * (1/sum(vimp[i, j, :]))

    print("counter equals {}".format(counter))
    print("counter 1 equals {}".format(counter1))
    print("counter 2 equals {}".format(counter2))

    return vimp

def Classification_vimp(vimp, predictions, y,):
    nsamples, nfeatures, c = vimp.shape
    vimp2 = np.zeros((nsamples, nfeatures))
    for i in range(0,nsamples):
        for j in range(0,nfeatures):
            #vimp2[i,j] = sum(vimp[i,j,:]) - vimp[i,j,y[i]]
            vimp2[i, j] = (1 - predictions[i][y[i]]) - (1 - vimp[i, j, y[i]])
    return vimp2