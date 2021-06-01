from experiments.ood_generalization.dataset import create_generalization_loaders

class GenBaseClients:
    def __init__(self, data_name, data_path, n_clients, n_gen_clients, batch_size=128, alpha=1, **kwargs):

        self.data_name = data_name
        self.data_path = data_path
        self.n_clients = n_clients
        self.n_gen_nodes = n_gen_clients
        self.alpha = alpha

        self.batch_size = batch_size

        self.train_loaders, self.val_loaders, self.test_loaders = None, None, None
        n_train_users = self.n_clients - self.n_gen_nodes
        self.train_loaders, self.val_loaders, self.test_loaders = create_generalization_loaders(
            self.data_name,
            self.data_path,
            n_train_users,
            self.n_gen_nodes,
            self.batch_size,
            self.alpha
        )

    def __len__(self):
        return self.n_clients

