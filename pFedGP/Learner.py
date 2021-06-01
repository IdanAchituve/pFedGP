import torch.nn as nn
from pFedGP.tree import BinaryTreepFedGPFull, BinaryTreepFedGPIPData, \
                        BinaryTreepFedGPIPCompute
from utils import *

class Model(nn.Module):
    ###########################
    # for cifar/imagenet need to use thumbnail version
    ############################
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.tree = None
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y, to_print=True, *args, **kwargs):
        raise NotImplementedError("not yet implemented")

    def forward_eval(self, x, y, is_first_iter, *args, **kwargs):
        raise NotImplementedError("not yet implemented")

    def build_base_tree(self, X, Y, *args, **kwargs):
        raise NotImplementedError("not yet implemented")


class pFedGPFullLearner(Model):
    def __init__(self, args, n_output=2):
        super(pFedGPFullLearner, self).__init__(args)
        self.n_output = n_output

    def forward(self, z, y, to_print=True):
        loss = self.tree.train_tree(z, y, to_print)
        return loss

    def forward_eval(self, X_train, Y_train, X_test, Y_test, is_first_iter=False):
        preds = self.tree.eval_tree_full_path(X_train, Y_train, X_test, self.n_output, is_first_iter)
        loss = CE_loss(Y_test, preds, self.n_output)

        return loss, preds

    def build_base_tree(self, X, Y):

        # Create tree instance
        self.tree = BinaryTreepFedGPFull(self.args, X.device)
        subtree_gp_counter = self.tree.build_tree(self.tree.root, X, Y)
        self.tree.to(X.device)
        return subtree_gp_counter


class pFedGPIPDataLearner(Model):
    def __init__(self, args, n_output=2):
        super(pFedGPIPDataLearner, self).__init__(args)
        self.n_output = n_output

    def forward(self, z, y, X_bar, to_print=True):
        loss = self.tree.train_tree(z, y, X_bar, to_print)
        return loss

    def forward_eval(self, X_train, Y_train, X_test, Y_test, X_bar, is_first_iter=False):
        preds = self.tree.eval_tree_full_path(X_train, Y_train, X_test, X_bar, self.n_output, is_first_iter)
        loss = CE_loss(Y_test, preds, self.n_output)

        return loss, preds

    def build_base_tree(self, X, Y, X_bar):
        # Create tree instance
        self.tree = BinaryTreepFedGPIPData(self.args, X.device)
        subtree_gp_counter = self.tree.build_tree(self.tree.root, X, Y, X_bar)
        self.tree.to(X.device)
        return subtree_gp_counter


class pFedGPIPComputeLearner(pFedGPIPDataLearner):

    def build_base_tree(self, X, Y, X_bar):

        # Create tree instance
        self.tree = BinaryTreepFedGPIPCompute(self.args, X.device)
        subtree_gp_counter = self.tree.build_tree(self.tree.root, X, Y)
        self.tree.to(X.device)
        return subtree_gp_counter

    def forward_eval(self, X_train, Y_train, X_test, Y_test, X_bar, is_first_iter=False):
        preds = self.tree.eval_tree_full_path(X_train, Y_train, X_test, X_bar, self.n_output, is_first_iter)
        loss = CE_loss(Y_test, preds, self.n_output)
        return loss, preds