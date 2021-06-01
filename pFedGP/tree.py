from pFedGP.node import NodepFedGPFull, NodepFedGPIPData, NodepFedGPIPCompute
from pFedGP.class_splits import *
from utils import (detach_to_numpy, pytorch_take)
import logging
from torch import nn
import torch
from collections import deque
import copy


class BinaryTree(nn.Module):
    def __init__(self, args, device):
        super(BinaryTree, self).__init__()
        self.root = None
        self.args = args
        self.device = device
        self.map_orig_to_tree_lbls = {}

    def split_func(self, X, Y, *args, **kwargs):
        # method for splitting classes
        return {'Split': Split(Y, 2),
                'MeanSplitKmeans': MeanSplitKmeans(Y, 2, X)}

    def get_root(self):
        return self.root

    def label_leaves(self):
        """
        Label leaves according to in-order scan.
        These label corresponds to labels given when doing evaluation
        """
        leaf_class = 0
        if self.root is not None:
            self._label_leaves(self.root, leaf_class)

    def _label_leaves(self, node, leaf_class):
        if node is not None:
            leaf_class = self._label_leaves(node.left_child, leaf_class)
            logging.info(str(node.classes) + ' ')
            if node.classes.size(0) == 1:
                self.map_orig_to_tree_lbls[node.classes.item()] = leaf_class
                leaf_class += 1
            leaf_class = self._label_leaves(node.right_child, leaf_class)
        return leaf_class


class BinaryTreepFedGPFull(BinaryTree):
    def __init__(self, args, device):
        super(BinaryTreepFedGPFull, self).__init__(args, device)
        self.root = NodepFedGPFull()
        self.root.id = 0
        self.root.depth = 0

    def build_tree(self, root, X, Y):
        """
        Build binary tree with GP attached to each node
        """
        # root
        q = deque()

        # push source vertex into the queue
        q.append((root, X, Y))
        curr_id = 1
        gp_counter = 0  # for getting avg. loss over the whole tree

        # loop till queue is empty
        while q:
            # pop front node from queue
            root, root_X, root_Y = q.popleft()
            node_classes, _ = torch.sort(torch.unique(root_Y))
            num_classes = node_classes.size(0)

            # two classes or less - no heuristic for splitting
            split_method = 'MeanSplitKmeans' if num_classes > 2 else 'Split'
            root_old_to_new = \
                self.split_func(detach_to_numpy(root_X),
                                detach_to_numpy(root_Y))[split_method].split()

            root.set_data(root_Y, root_old_to_new)

            # leaf node
            if num_classes == 1:
                # logging.info('Reached a leaf node. Node index: ' + str(root.id) + ' ')
                continue

            # Internal node
            else:
                gp_counter += 1
                root.set_model(self.args.kernel_function,
                               self.args.num_gibbs_steps_train, self.args.num_gibbs_draws_train,
                               self.args.num_gibbs_steps_test, self.args.num_gibbs_draws_test,
                               self.args.outputscale_increase, self.args.outputscale,
                               self.args.lengthscale, self.args.predict_ratio, self.args.objective)

                left_X, left_Y = pytorch_take(root_X, root_Y, root.new_to_old[0])
                right_X, right_Y = pytorch_take(root_X, root_Y, root.new_to_old[1])
                child_X = [left_X, right_X]
                child_Y = [left_Y, right_Y]

                branches = 2
                for i in range(branches):
                    child = NodepFedGPFull()
                    child.id = curr_id
                    curr_id += 1
                    child.depth = root.depth + 1
                    root.set_child(child, i)
                    q.append((child, child_X[i], child_Y[i]))

        return gp_counter

    def train_tree(self, X, Y, to_print=True):
        loss = 0
        if self.root is not None:
            loss = self._train_tree(self.root, X, Y, to_print)
        return loss

    def _train_tree(self, node, X, Y, to_print=True):

        loss = 0
        node_classes = set(detach_to_numpy(node.classes).tolist())
        batch_classes = set(detach_to_numpy(Y).tolist())

        # enter if it is an internal node and there are at least 1 example in the batch for that node
        if node.classes.size(0) > 1 and len(batch_classes.intersection(node_classes)) > 0:
            loss += self._train_tree(node.left_child, X, Y, to_print)
            if to_print:
                logging.info('Training GP on classes: ' + str(detach_to_numpy(node.classes).tolist()) + ' ')
            node_X, node_Y = self._extract_node_data(node, X, Y)
            loss += node.train_loop(node_X, node_Y, to_print)
            loss += self._train_tree(node.right_child, X, Y, to_print)
        else:
            if to_print:
                logging.info('No need for training. Class: ' + str(detach_to_numpy(node.classes).tolist()) + ' ')
        return loss

    def _extract_node_data(self, node, X, Y):
        # take data belongs to that node only
        node_X, node_Y_orig = pytorch_take(X, Y, node.classes)
        # from original labels to node labels
        node_Y = torch.tensor([node.old_to_new[y.item()] for y in node_Y_orig], dtype=Y.dtype).to(Y.device)
        return node_X, node_Y

    def eval_tree_full_path(self, X_train, Y_train, X_test, num_classes, is_first_iter=False):
        # create a queue used to do BFS
        q = deque()

        # accumulated log probability matrix
        probs_mat = torch.ones((X_test.shape[0], num_classes), dtype=X_test.dtype, device=X_test.device)

        # push source vertex into the queue
        q.append(self.root)

        # loop till queue is empty
        while q:
            # pop front node from queue and print it
            node = q.popleft()

            # In case of only one class, all predictions are of that class. Nothing to add to the queue
            if node.classes.size(0) == 1:
                #logging.info('No need for evaluation. Class: ' + str(node.classes) + ' ')
                continue

            # In case more than one class run GP on the node
            else:
                node_X, node_Y = self._extract_node_data(node, X_train, Y_train)

                probs = node.model.predictive_posterior(X=node_X, Y=node_Y, X_star=X_test, is_first_iter=is_first_iter)

                left_classes = node.new_to_old[0]
                right_classes = node.new_to_old[1]

                probs = probs.unsqueeze(1)
                class_probs = torch.cat((probs, 1 - probs), dim=1)

                probs_mat[:, left_classes] = probs_mat[:, left_classes] * class_probs[:, 0].reshape(-1, 1)
                probs_mat[:, right_classes] = probs_mat[:, right_classes] * class_probs[:, 1].reshape(-1, 1)

                # more than 2 children - not a leaf node. Add child nodes to queue.
                if node.classes.size(0) > 2:
                    q.append(node.left_child)
                    q.append(node.right_child)

        return probs_mat


class BinaryTreepFedGPIPData(BinaryTree):

    def __init__(self, args, device):
        super(BinaryTreepFedGPIPData, self).__init__(args, device)
        self.root = NodepFedGPIPData()
        self.root.id = 0
        self.root.depth = 0

    def build_tree(self, root, X, Y, X_bar):
        """
        Build binary tree with GP attached to each node
        """
        # root
        q = deque()

        # push source vertex into the queue
        q.append((root, X, Y))
        curr_id = 1
        gp_counter = 0  # for getting avg. loss over the whole tree

        # loop till queue is empty
        while q:
            # pop front node from queue
            root, root_X, root_Y = q.popleft()
            node_classes, _ = torch.sort(torch.unique(root_Y))
            num_classes = node_classes.size(0)

            # Xbar's of current node
            X_bar_root = X_bar[node_classes, ...]

            # two classes or less - no heuristic for splitting
            split_method = 'MeanSplitKmeans' if num_classes > 2 else 'Split'
            root_old_to_new = \
                self.split_func(detach_to_numpy(root_X),
                                detach_to_numpy(root_Y))[split_method].split()

            root.set_data(root_Y, root_old_to_new)

            # build label vector of current node
            num_Xbars = X_bar_root.shape[1]
            i = 0
            for original_lbl, node_lbl in root_old_to_new.items():
                Y_bar_class = torch.zeros(num_Xbars, device=Y.device, dtype=Y.dtype) if node_lbl == 0 \
                    else torch.ones(num_Xbars, device=Y.device, dtype=Y.dtype)
                Y_bar_root = Y_bar_class if i == 0 else torch.cat((Y_bar_root, Y_bar_class))
                i += 1

            # leaf node
            if num_classes == 1:
                # logging.info('Reached a leaf node. Node index: ' + str(root.id) + ' ')
                continue

            # Internal node
            else:
                gp_counter += 1
                root.set_model(self.args.kernel_function,
                               self.args.num_gibbs_steps_train, self.args.num_gibbs_draws_train,
                               self.args.num_gibbs_steps_test, self.args.num_gibbs_draws_test,
                               self.args.outputscale_increase, self.args.outputscale,
                               self.args.lengthscale, Y_bar_root, self.args.balance_classes)

                left_X, left_Y = pytorch_take(root_X, root_Y, root.new_to_old[0])
                right_X, right_Y = pytorch_take(root_X, root_Y, root.new_to_old[1])
                child_X = [left_X, right_X]
                child_Y = [left_Y, right_Y]

                branches = 2
                for i in range(branches):
                    child = NodepFedGPIPData()
                    child.id = curr_id
                    curr_id += 1
                    child.depth = root.depth + 1
                    root.set_child(child, i)
                    q.append((child, child_X[i], child_Y[i]))

        return gp_counter

    def train_tree(self, X, Y, X_bar, to_print=True):
        loss = 0
        if self.root is not None:
            loss = self._train_tree(self.root, X, Y, X_bar, to_print)
        return loss

    def _train_tree(self, node, X, Y, X_bar, to_print=True):

        loss = 0
        node_classes = set(detach_to_numpy(node.classes).tolist())
        batch_classes = set(detach_to_numpy(Y).tolist())

        # enter if it is an internal node and there are at least 1 example in the batch for that node
        if node.classes.size(0) > 1 and len(batch_classes.intersection(node_classes)) > 0:
            loss += self._train_tree(node.left_child, X, Y, X_bar, to_print)
            if to_print:
                logging.info('Training GP on classes: ' + str(detach_to_numpy(node.classes).tolist()) + ' ')
            node_X, node_Y, node_Xbar = self._extract_node_data(node, X, Y, X_bar)
            loss += node.train_loop(node_X, node_Y, node_Xbar, to_print)
            loss += self._train_tree(node.right_child, X, Y, X_bar, to_print)
        else:
            if to_print:
                logging.info('No need for training. Class: ' + str(detach_to_numpy(node.classes).tolist()) + ' ')
        return loss

    def _extract_node_data(self, node, X, Y, X_bar):
        # take data belongs to that node only
        node_X, node_Y_orig = pytorch_take(X, Y, node.classes)
        # from original labels to node labels
        node_Y = torch.tensor([node.old_to_new[y.item()] for y in node_Y_orig], dtype=Y.dtype).to(Y.device)
        node_Xbar = X_bar[node.classes, ...]
        return node_X, node_Y, node_Xbar

    def eval_tree_full_path(self, X_train, Y_train, X_test, X_bar, num_classes, is_first_iter=False):
        # create a queue used to do BFS
        q = deque()

        # accumulated log probability matrix
        probs_mat = torch.ones((X_test.shape[0], num_classes), dtype=X_test.dtype, device=X_test.device)

        # push source vertex into the queue
        q.append(self.root)

        # loop till queue is empty
        while q:
            # pop front node from queue and print it
            node = q.popleft()

            # In case of only one class, all predictions are of that class. Nothing to add to the queue
            if node.classes.size(0) == 1:
                #logging.info('No need for evaluation. Class: ' + str(node.classes) + ' ')
                continue

            # In case more than one class run GP on the node
            else:

                node_X, node_Y, node_Xbar = self._extract_node_data(node, X_train, Y_train, X_bar)
                nodeXbar = node_Xbar.reshape(node_Xbar.shape[0] * node_Xbar.shape[1], -1)
                nodeXtrain = torch.cat((node_X, nodeXbar), dim=0)
                nodeYtrain = torch.cat((node_Y, node.Y_bar), dim=0)

                probs = node.model.predictive_posterior(X_star=X_test, X=nodeXtrain, Y=nodeYtrain,
                                                        is_first_iter=is_first_iter)

                left_classes = node.new_to_old[0]
                right_classes = node.new_to_old[1]

                probs = probs.unsqueeze(1)
                class_probs = torch.cat((probs, 1 - probs), dim=1)
                if self.args.balance_classes:
                    node_labels, node_label_counts = torch.unique(node_Y, return_counts=True)
                    # P(Y_* | X_*) ∝ (P(Y)/Q(Y)) * Q(Y_*|X_*); where P(Y) is unbalanced and Q(Y) is balanced
                    p_y_s0 = node_label_counts[0] / (node_label_counts[0] + node_label_counts[1])
                    p_y_s1 = node_label_counts[1] / (node_label_counts[0] + node_label_counts[1])
                    q_y_s0 = (node_label_counts[0] + torch.sum(node.Y_bar == 0)) / nodeYtrain.shape[0]
                    q_y_s1 = (node_label_counts[1] + torch.sum(node.Y_bar == 1)) / nodeYtrain.shape[0]

                    class_probs[:, 0] *= (p_y_s0 / q_y_s0)
                    class_probs[:, 1] *= (p_y_s1 / q_y_s1)
                    # normalize
                    class_probs /= torch.sum(class_probs, dim=1, keepdim=True)

                probs_mat[:, left_classes] = probs_mat[:, left_classes] * class_probs[:, 0].reshape(-1, 1)
                probs_mat[:, right_classes] = probs_mat[:, right_classes] * class_probs[:, 1].reshape(-1, 1)

                # more than 2 children - not a leaf node. Add child nodes to queue.
                if node.classes.size(0) > 2:
                    q.append(node.left_child)
                    q.append(node.right_child)

        return probs_mat


class BinaryTreepFedGPIPCompute(BinaryTreepFedGPIPData):
    def __init__(self, args, device):
        super(BinaryTreepFedGPIPCompute, self).__init__(args, device)
        self.root = NodepFedGPIPCompute()
        self.root.id = 0
        self.root.depth = 0

    def build_tree(self, root, X, Y, X_bar=None):
        """
        Build binary tree with GP attached to each node
        """
        # root
        q = deque()

        # push source vertex into the queue
        q.append((root, X, Y))
        curr_id = 1
        gp_counter = 0  # for getting avg. loss over the whole tree

        # loop till queue is empty
        while q:
            # pop front node from queue
            root, root_X, root_Y = q.popleft()
            node_classes, _ = torch.sort(torch.unique(root_Y))
            num_classes = node_classes.size(0)

            # two classes or less - no heuristic for splitting
            split_method = 'MeanSplitKmeans' if num_classes > 2 else 'Split'
            root_old_to_new = \
                self.split_func(detach_to_numpy(root_X),
                                detach_to_numpy(root_Y))[split_method].split()

            root.set_data(root_Y, root_old_to_new)

            # leaf node
            if num_classes == 1:
                # logging.info('Reached a leaf node. Node index: ' + str(root.id) + ' ')
                continue

            # Internal node
            else:
                gp_counter += 1
                root.set_model(self.args.kernel_function,
                               self.args.num_gibbs_steps_train, self.args.num_gibbs_draws_train,
                               self.args.num_gibbs_steps_test, self.args.num_gibbs_draws_test,
                               self.args.outputscale_increase, self.args.outputscale,
                               self.args.lengthscale, self.args.predict_ratio, self.args.objective)

                left_X, left_Y = pytorch_take(root_X, root_Y, root.new_to_old[0])
                right_X, right_Y = pytorch_take(root_X, root_Y, root.new_to_old[1])
                child_X = [left_X, right_X]
                child_Y = [left_Y, right_Y]

                branches = 2
                for i in range(branches):
                    child = NodepFedGPIPCompute()
                    child.id = curr_id
                    curr_id += 1
                    child.depth = root.depth + 1
                    root.set_child(child, i)
                    q.append((child, child_X[i], child_Y[i]))

        return gp_counter

    def eval_tree_full_path(self, X_train, Y_train, X_test, X_bar, num_classes, is_first_iter=False):
        # create a queue used to do BFS
        q = deque()

        # accumulated log probability matrix
        probs_mat = torch.ones((X_test.shape[0], num_classes), dtype=X_test.dtype, device=X_test.device)

        # push source vertex into the queue
        q.append(self.root)

        # loop till queue is empty
        while q:
            # pop front node from queue and print it
            node = q.popleft()

            # In case of only one class, all predictions are of that class. Nothing to add to the queue
            if node.classes.size(0) == 1:
                # logging.info('No need for evaluation. Class: ' + str(node.classes) + ' ')
                continue

            # In case more than one class run GP on the node
            else:
                node_X, node_Y, node_Xbar = self._extract_node_data(node, X_train, Y_train, X_bar)

                X_bar_root = node_Xbar.reshape(X_bar[node.classes, ...].shape[0] * X_bar.shape[1], -1)
                probs = node.model.predictive_posterior(X=node_X, Y=node_Y, X_star=X_test, X_bar=X_bar_root,
                                                        is_first_iter=is_first_iter)

                left_classes = node.new_to_old[0]
                right_classes = node.new_to_old[1]

                probs = probs.unsqueeze(1)
                class_probs = torch.cat((probs, 1 - probs), dim=1)

                probs_mat[:, left_classes] = probs_mat[:, left_classes] * class_probs[:, 0].reshape(-1, 1)
                probs_mat[:, right_classes] = probs_mat[:, right_classes] * class_probs[:, 1].reshape(-1, 1)

                # more than 2 children - not a leaf node. Add child nodes to queue.
                if node.classes.size(0) > 2:
                    q.append(node.left_child)
                    q.append(node.right_child)

        return probs_mat
