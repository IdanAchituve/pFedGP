import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class NoisyClients:
    # todo: for now the only transformation on images is / 255.
    def __init__(
            self,
            data_path,
            batch_size=128,
    ):
        with open(data_path, "rb") as f:
            self.data_dict = pickle.load(f)

        # create data loaders
        self.train_loaders, self.val_loaders, self.test_loaders = [], [], []
        for client_id, data in self.data_dict.items():
            self.train_loaders.append(
                DataLoader(
                    TensorDataset(
                        torch.from_numpy(data['train']['data'].astype(np.float32) / 255.),
                        torch.from_numpy(data['train']['label'])
                    ),
                    shuffle=True,
                    batch_size=batch_size
                )

            )

            self.val_loaders.append(
                DataLoader(
                    TensorDataset(
                        torch.from_numpy(data['val']['data'].astype(np.float32) / 255.),
                        torch.from_numpy(data['val']['label'])
                    ),
                    shuffle=False,
                    batch_size=batch_size
                )

            )

            self.test_loaders.append(
                DataLoader(
                    TensorDataset(
                        torch.from_numpy(data['test']['data'].astype(np.float32) / 255.),
                        torch.from_numpy(data['test']['label'])
                    ),
                    shuffle=False,
                    batch_size=batch_size
                )

            )

        # local layers
        self.n_clients = len(self.data_dict)

    def __len__(self):
        return self.n_clients