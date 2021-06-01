import numpy as np
#from experiments.cub_pretrained.hierarchical.minmax_kmeans import minsize_kmeans, compute_quality
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans


class Split(object):
    def __init__(self, labels, branches=3):
        self.old_to_new = {}
        self.labels = labels
        self.classes = np.unique(labels)
        self.num_classes = self.classes.shape[0]
        self.branches = branches

    def split(self, *args, **kwargs):
        if self.num_classes == 3:
            self.old_to_new[self.classes[0]] = 0
            self.old_to_new[self.classes[1]] = 1
            self.old_to_new[self.classes[2]] = 2
        elif self.num_classes == 2:
            self.old_to_new[self.classes[0]] = 0
            self.old_to_new[self.classes[1]] = 1
        else:
            self.old_to_new[self.classes[0]] = 0
        return self.old_to_new


class RandomSplit(Split):
    """
    split labels associated with a node to x branches randomly
    :param labels: numpy array of labels
    :param branches: the number of branches
    :return the original classes partitioned to nodes
    """
    def __init__(self, labels, branches):
        super().__init__(labels, branches)

    def split(self):
        #if self.num_classes > 3:  # currently use this hard threshold
        classes_perm = np.random.permutation(self.classes)
        for i in range(self.branches):
            lower = i * (self.num_classes // self.branches)
            upper = lower + np.floor(self.num_classes / self.branches) if i != self.branches - 1 else \
                lower + np.ceil(self.num_classes / self.branches)
            node_classes = classes_perm[lower:int(upper)]
            self.old_to_new.update({o: i for o in node_classes.tolist()})

        return self.old_to_new


class MeanSplitAgglomerative(Split):
    """
    split labels associated with a node to x branches by the mean vector of each class.
    close classes should be grouped together
    :param labels: numpy array of the labels
    :param branches: the number of branches
    :param data: numpy array of the data
    :param affinity: Metric - “euclidean”, “l1”, “l2”, “manhattan”, “cosine”
    :param linkage: Distance to use between sets of observation: “ward”, “complete”, “average”, “single”
    :return the original classes partitioned to nodes
    """
    def __init__(self, labels, branches, data, affinity='euclidean', linkage='ward'):
        super().__init__(labels, branches)
        self.affinity = affinity
        self.linkage = linkage
        self.data = data

    def split(self):

        # mean vector of each class
        means = np.array([0])
        for idx, i in enumerate(self.classes):
            tmp = self.data[np.where(self.labels == i)]
            mean_vec = np.mean(tmp, axis=0, keepdims=True)
            means = mean_vec if idx == 0 else np.concatenate((means, mean_vec), axis=0)

        # hierarchical clustreing
        n_clusters = self.branches
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity=self.affinity, linkage=self.linkage)
        lbl_assignment = clustering.fit(means).labels_

        for o, n in zip(self.classes, lbl_assignment):
            self.old_to_new.update({o.item(): n.item()})

        return self.old_to_new


class MeanSplitKmeans(Split):
    """
    split labels associated with a node to x branches by the mean vector of each class.
    close classes should be grouped together
    :param labels: numpy array of the labels
    :param branches: the number of branches
    :param data: numpy array of the data
    :param affinity: Metric - “euclidean”, “cosine”
    :param linkage: Distance to use between sets of observation: “ward”, “complete”, “average”, “single”
    :return the original classes partitioned to nodes
    """
    def __init__(self, labels, branches, data, affinity='euclidean'):
        super().__init__(labels, branches)
        self.affinity = affinity
        self.data = data

    def split(self):

        # mean vector of each class
        means = np.array([0])
        for idx, i in enumerate(self.classes):
            tmp = self.data[np.where(self.labels == i)]
            mean_vec = np.mean(tmp, axis=0, keepdims=True)
            if self.affinity == 'cosine':
                mean_vec /= np.linalg.norm(mean_vec)
            means = mean_vec if idx == 0 else np.concatenate((means, mean_vec), axis=0)

        # hierarchical clustreing
        n_clusters = self.branches
        clustering = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
        lbl_assignment = clustering.fit(means).labels_

        for o, n in zip(self.classes, lbl_assignment):
            self.old_to_new.update({o.item(): n.item()})

        return self.old_to_new

class ProtoTypeSplit(Split):
    """
    split labels associated with a node to x branches by the prototype of each class.
    close classes should be grouped together
    :param labels: numpy array of the labels
    :param branches: the number of branches
    :param prototype: dictionary of {label: np.array()}
    :param affinity: Metric - “euclidean”, “l1”, “l2”, “manhattan”, “cosine”
    :param linkage: Distance to use between sets of observation: “ward”, “complete”, “average”, “single”
    :return the original classes partitioned to nodes
    """
    def __init__(self, labels, branches, prototype, affinity='cosine', linkage='complete'):
        super().__init__(labels, branches)
        self.affinity = affinity
        self.linkage = linkage
        self.prototype = prototype

    def split(self):

        # hierarchical clustreing
        n_clusters = self.branches
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity=self.affinity, linkage=self.linkage)
        lbl_assignment = clustering.fit(list(self.prototype.values())).labels_

        for o, n in zip(self.prototype.keys(), lbl_assignment):
            self.old_to_new.update({o: n.item()})

        return self.old_to_new