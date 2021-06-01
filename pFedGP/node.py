from pFedGP.pFedGP_full_data import pFedGPFull, pFedGPFullBound, pFedGPIPData
from pFedGP.pFedGP_compute import pFedGPIPCompute
from torch import nn
from utils import *


class Node(nn.Module):
    def __init__(self):
        super(Node, self).__init__()
        self.left_child = None
        self.right_child = None
        self.model = None
        self.device = None
        self.id = 0
        self.depth = 0
        self.init_node()

    def init_node(self):
        self.old_to_new = {}
        self.new_to_old = {}
        self.classes = []

    def set_child(self, node, child=0):
        if child == 0:  # left child
            self.left_child = node
        elif child == 1:  # right child
            self.right_child = node
        else:
            raise NotImplementedError("not a valid child")

    def map_old_to_new_lbls(self, Y):
        new_to_old = {}
        new_lbls = torch.clone(Y)
        for old_class, new_class in self.old_to_new.items():
            if new_class in new_to_old:  # if key exists append to the list of corresponding old classes
                new_to_old[new_class] = new_to_old[new_class] + [old_class]
            else:
                new_to_old[new_class] = [old_class]
            # assign new label. add constant to prevent from running over new class in following iterations
            new_lbls[new_lbls == old_class] = new_class + 1000

        new_lbls -= 1000
        return new_lbls, new_to_old


class NodepFedGPFull(Node):

    def set_data(self, Y_support, old_to_new):
        self.classes, _ = torch.sort(torch.unique(Y_support))
        self.old_to_new = old_to_new

        Y_supp_new_lbls, new_to_old = self.map_old_to_new_lbls(Y_support)
        self.new_to_old = new_to_old

    def set_model(self, kernel_function, num_steps, num_draws,
                  num_steps_test, num_draws_test,
                  outputscale_increase, outputscale, lengthscale,
                  predict_ratio, objective='predictive_likelihood'):

        num_classes = 2
        self.model = pFedGPFull(kernel_func=kernel_function, num_classes=num_classes,
                                num_steps=num_steps, num_draws=num_draws,
                                num_steps_test=num_steps_test, num_draws_test=num_draws_test,
                                predict_ratio=predict_ratio)

        depth = 0 if outputscale_increase == 'constant' else self.depth \
                if outputscale_increase == 'increase' else - self.depth

        self.model.model._set_params(outputscale=max(outputscale + depth, 0.1),
                                     lengthscale=lengthscale + depth * 0.1)

        self.objective = objective
        self.model.to(self.device)

    def train_loop(self, X, Y, to_print=True, *args, **kwargs):

        objective = self.model.forward_predictive if self.objective == 'predictive_likelihood' \
                    else self.model.forward_mll
        loss = objective(X, Y, to_print=to_print)
        return loss


class NodepFedGPIPData(Node):

    def set_data(self, Y_support, old_to_new):
        self.classes, _ = torch.sort(torch.unique(Y_support))
        self.old_to_new = old_to_new

        Y_supp_new_lbls, new_to_old = self.map_old_to_new_lbls(Y_support)
        self.new_to_old = new_to_old


    def set_model(self, kernel_function, num_steps, num_draws,
                  num_steps_test, num_draws_test,
                  outputscale_increase, outputscale, lengthscale,
                  Y_bar, balance_classes):

        self.Y_bar = Y_bar

        num_classes = 2
        self.model = pFedGPIPData(kernel_func=kernel_function, num_classes=num_classes,
                                  num_steps=num_steps, num_draws=num_draws,
                                  num_steps_test=num_steps_test, num_draws_test=num_draws_test,
                                  balance_classes=balance_classes)

        depth = 0 if outputscale_increase == 'constant' else self.depth \
                if outputscale_increase == 'increase' else - self.depth

        self.model.model._set_params(outputscale=max(outputscale + depth, 0.1),
                                     lengthscale=lengthscale + depth * 0.1)

        self.model.to(self.device)

    def train_loop(self, X, Y, X_bar=None, to_print=True):
        loss = self.model.forward_predicitive(X, Y, X_bar, self.Y_bar, to_print=to_print)
        return loss


class NodepFedGPIPCompute(NodepFedGPFull):

    def set_model(self, kernel_function, num_steps, num_draws,
                  num_steps_test, num_draws_test,
                  outputscale_increase, outputscale, lengthscale,
                  predict_ratio, objective='predictive_likelihood'):

        num_classes = 2
        self.model = pFedGPIPCompute(kernel_func=kernel_function, num_classes=num_classes,
                                     num_steps=num_steps, num_draws=num_draws,
                                     num_steps_test=num_steps_test, num_draws_test=num_draws_test,
                                     predict_ratio=predict_ratio)

        depth = 0 if outputscale_increase == 'constant' else self.depth \
                if outputscale_increase == 'increase' else - self.depth

        self.model.model._set_params(outputscale=max(outputscale + depth, 0.1),
                                     lengthscale=lengthscale + depth * 0.1)

        self.objective = objective
        self.model.to(self.device)

    def train_loop(self, X, Y, X_bar=None, to_print=True, *args, **kwargs):
        objective = self.model.forward_predictive if self.objective == 'predictive_likelihood' \
            else self.model.forward_mll
        loss = objective(X, Y, X_bar, to_print=to_print)
        return loss