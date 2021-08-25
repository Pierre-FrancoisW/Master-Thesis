import numpy as np

def prob_node_left(tree, node):
    # [tree.children_left[node]] = node + 1 as long as node is not a leaf
    return tree.n_node_samples[tree.children_left[node]] / tree.n_node_samples[node]

def exploreC(tree, node, children_right_view, sample):

    if tree.feature[node] == -2:
        value = tree.value[node].ravel() / tree.value[node].max()
        return value
    # else check current split and define next direction
    else:
        if sample[tree.feature[node]] <= tree.threshold[node]:
            # left child is next node
            return exploreC(tree, node+1, children_right_view, sample)
        else:
            return exploreC(tree, children_right_view[node], children_right_view, sample)


def weighted_prediction_C(tree, prediction, nodes_id, leaf, children_right_view, sample):
    dir = 'left'
    node = nodes_id[0]
    # direction of sample at split
    if leaf >= children_right_view[node]:
        dir = 'right'

    l_prob = prob_node_left(tree, nodes_id[0])
    r_prob = 1 - l_prob

    if l_prob > 1:
        print("Error lprob")
    if r_prob > 1:
        print("Error rightprob")

    # print(l_prob, "  ", r_prob)
    # print(dir)
    if nodes_id.size == 1:
        if dir == 'left':
            value = (prediction * l_prob) + (r_prob * exploreC(tree, children_right_view[node], children_right_view, sample))
            return (prediction * l_prob) + (r_prob * exploreC(tree, children_right_view[node], children_right_view, sample))
        else:
            value = (prediction * r_prob) + (l_prob * exploreC(tree, node + 1, children_right_view, sample))
            return (prediction * r_prob) + (l_prob * exploreC(tree, node + 1, children_right_view, sample))
    nodes_id = nodes_id[1:]
    if dir == 'left':
        value1 = r_prob * exploreC(tree, children_right_view[node], children_right_view, sample)
        value2 = l_prob * weighted_prediction_C(tree, prediction, nodes_id, leaf, children_right_view, sample)
        return (r_prob * exploreC(tree, children_right_view[node], children_right_view, sample)) + (l_prob * weighted_prediction_C(tree, prediction, nodes_id, leaf, children_right_view, sample))
    else:
        value = (l_prob * exploreC(tree, node + 1, children_right_view, sample)) + (r_prob * weighted_prediction_C(tree, prediction, nodes_id, leaf, children_right_view, sample))
        return (l_prob * exploreC(tree, node + 1, children_right_view, sample)) + (r_prob * weighted_prediction_C(tree, prediction, nodes_id, leaf, children_right_view, sample))


def compute_mda_local_treeC(Ctree, X, nsamples, nfeatures,nclass, vimp):
    # use Ctree.value[node]
    children_left_view = Ctree.children_left
    children_right_view = Ctree.children_right
    feature_view = Ctree.feature
    feature_range = list(range(nfeatures))
    node_indicator = Ctree.decision_path(X)
    node = 0
    ifeat = 0

    for i in range(nsamples):
        # features of decision path
        features = feature_view[node_indicator[i, :].indices][:-1]
        unique_f = np.unique(features)
        # node id of decision path
        nodes = node_indicator[i, :].indices[:-1]
        prediction = Ctree.predict(X)[i].ravel()
        prediction = prediction/ np.max(prediction)
        leaf = node_indicator[i, :].indices[-1]
        for x in unique_f:
            nodes_id = nodes[features == x]
            val = weighted_prediction_C(Ctree, prediction, nodes_id, leaf, children_right_view, X[i])
            vimp[i, x, :] = vimp[i, x, :] + val
    return vimp


def compute_mda_local_ens_C(ens, X):
    nsamples = X.shape[0]
    nfeatures = X.shape[1]
    nclass = ens.classes_.size
    vimp = [[[0.0 for x in range(nclass)] for y in range(nfeatures)] for x in range(nsamples)]
    vimp = np.array(vimp)
    # Dim1 : samples, Dim2: feature, Dim3: class
    nestimators = ens.n_estimators

    for i in range(nestimators):
        # print("o", end='', flush=True)
        vimp = compute_mda_local_treeC(ens.estimators_[i].tree_, X, nsamples, nfeatures,nclass, vimp)

    print("")
    vimp /= (ens.n_estimators)

    for i in range(0,nsamples):
        for j in range(0,nfeatures):
            vimp[i, j, :] = vimp[i, j, :] * (1/sum(vimp[i, j, :]))

    return vimp