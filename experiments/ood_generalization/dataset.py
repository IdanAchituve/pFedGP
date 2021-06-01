from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10
from sklearn.model_selection import train_test_split


def classes_per_node_dirichlet(labels_list, num_users, alpha):
    if isinstance(labels_list, torch.Tensor):
        num_classes = len(labels_list.unique())
    else:
        num_classes = len(np.unique(labels_list))

    # create distribution for each client
    alpha_list = [alpha for _ in range(num_classes)]
    np.random.seed(42)
    prob_array = np.random.dirichlet(alpha_list, num_users)

    # normalizing
    prob_array /= prob_array.sum(axis=0)

    class_partitions = defaultdict(list)
    cls_list = [i for i in range(num_classes)]
    for i in range(num_users):
        class_partitions['class'].append(cls_list)
        class_partitions['prob'].append(prob_array[i, :])

    return class_partitions


def gen_data_split(labels_list, num_users, class_partitions):
    if isinstance(labels_list, torch.Tensor):
        labels_list = labels_list.cpu().numpy().astype(int)
    num_classes, num_samples = np.unique(labels_list, return_counts=True)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(labels_list == i)[0] for i in range(len(num_classes))}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        np.random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = [[] for _ in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx


def create_generalization_loaders(data_name, data_root, num_train_users, num_gen_users, bz, alpha: float = 10):
    # get datasets and idxs of each split
    train_dataset, test_dataset, train_idx, val_idx, test_idx = get_datasets(data_name, data_root)
    # create train / novel nodes partitions
    train_nodes_idx, novel_nodes_idx = idx_partition_per_group(
        train_idx, val_idx, test_idx, novel_nodes_size=float(num_gen_users / (num_train_users + num_gen_users))
    )

    # iterate over groups train/novel + different splits train/val/test
    idx_user_split = [[[] for _ in range(3)] for _ in range(2)]
    for g_id, g_nodes_split in enumerate((train_nodes_idx, novel_nodes_idx)):  # iterate over train / novel nodes
        # g_nodes_split holds train/novel train/val/test indexes
        n_users = num_train_users if g_id == 0 else num_gen_users
        for split_id, s in enumerate(g_nodes_split):  # iterate over train/val/test splits

            # assuming train/val/test order
            # check if split is test or not
            if split_id != 2:
                if isinstance(train_dataset.targets, list):
                    labels_list = np.array(train_dataset.targets)
                else:
                    labels_list = train_dataset.targets
                labels_list = labels_list[s]

            else:  # train / val case
                if isinstance(test_dataset.targets, list):
                    labels_list = np.array(test_dataset.targets)
                else:
                    labels_list = test_dataset.targets
                labels_list = labels_list[s]

            if split_id == 0:
                if g_id == 0:
                    class_partitions = classes_per_node_dirichlet(labels_list, num_users=n_users, alpha=alpha)
                else:
                    alpha_gen = alpha
                    class_partitions = classes_per_node_dirichlet(labels_list, num_users=n_users, alpha=alpha_gen)
            labels_list_index = gen_data_split(labels_list, n_users, class_partitions)
            idx_user_split[g_id][split_id].extend([np.array(s)[i] for i in labels_list_index])

    # unite groups and create dataloaders
    generalization_loaders = []
    # change order of clientes - first 10 are novel clients
    idx_user_split = idx_user_split[::-1]
    for s_i in range(len(idx_user_split[0])):
        loaders = []
        if s_i != 2:
            data = train_dataset
        else:
            data = test_dataset
        for g_i in range(len(idx_user_split)):
            for u_idx in idx_user_split[g_i][s_i]:
                loaders.append(DataLoader(Subset(data, u_idx), bz, (s_i == 0)))
        generalization_loaders.append(loaders)
    return generalization_loaders


def idx_partition_per_group(train_idx, val_idx, test_idx, novel_nodes_size=0.2):
    train_nodes_idx, novel_nodes_idx = [], []
    for id in (train_idx, val_idx, test_idx):
        t_nodes_idx, n_nodes_idx = train_test_split(range(len(id)), test_size=novel_nodes_size, random_state=42)
        train_nodes_idx.append(t_nodes_idx)
        novel_nodes_idx.append(n_nodes_idx)
    return train_nodes_idx, novel_nodes_idx


def get_datasets(data_name, dataroot, normalize=True, val_size=10000):

    if data_name == 'cifar10':
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        data_obj = CIFAR10
    elif data_name == 'cifar100':
        normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        data_obj = CIFAR100

    else:
        raise ValueError(f'data_name should be one of {["cifar10", "cifar100"]}')

    trans = [transforms.ToTensor()]

    if normalize:
        trans.append(normalization)

    transform = transforms.Compose(trans)

    train_set = data_obj(
        dataroot,
        train=True,
        download=True,
        transform=transform
    )

    test_set = data_obj(
        dataroot,
        train=False,
        download=True,
        transform=transform
    )

    train_size = len(train_set) - val_size
    train_idx, val_idx = train_test_split(range(len(train_set)), train_size=train_size, random_state=42)
    test_idx = list(range(len(test_set)))

    return train_set, test_set, train_idx, val_idx, test_idx
