from experiments.heterogeneous_class_dist.dataset import gen_random_loaders


class BaseClients:
    def __init__(
            self,
            data_name,
            data_path,
            n_clients,
            classes_per_client=2,
            batch_size=128
    ):

        self.data_name = data_name
        self.data_path = data_path
        self.n_clients = n_clients
        self.classes_per_client = classes_per_client

        self.batch_size = batch_size

        self.train_loaders, self.val_loaders, self.test_loaders = None, None, None
        self._init_dataloaders()

    def _init_dataloaders(self):
        self.train_loaders, self.val_loaders, self.test_loaders = gen_random_loaders(
            self.data_name,
            self.data_path,
            self.n_clients,
            self.batch_size,
            self.classes_per_client
        )

    def __len__(self):
        return self.n_clients