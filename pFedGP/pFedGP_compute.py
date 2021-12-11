from collections import namedtuple
import pypolyagamma
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from torch import nn
from pFedGP.kernel_class import OneClassGPModel
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

from utils import *

NodeGibbsState = namedtuple("NodeGibbsState", ["omega", "f"])
NodeModelState = namedtuple(
    "NodeModelState",
    ["Y", "M", "N", "N_sb", "kappa", "mu", "K", "L",
     "Knm_Kmminv_kmn", "Knm_Kmminv_mu", "Knn_Minus_Knm_Kmm_Kmn"],
)

class pFedGPIPCompute(nn.Module):
    def __init__(self,
                 kernel_func,
                 num_classes=2,
                 num_steps=10,
                 num_draws=20,
                 num_steps_test=10,
                 num_draws_test=30,
                 predict_ratio=0.4):

        super(pFedGPIPCompute, self).__init__()
        self.num_classes = num_classes

        self.model = OneClassGPModel(kernel_func)
        self.ppg = pypolyagamma.PyPolyaGamma()
        self.kernel_func = kernel_func

        self.num_steps_train = num_steps
        self.num_draws_train = num_draws
        self.num_steps_test = num_steps_test
        self.num_draws_test = num_draws_test
        self.predict_ratio = predict_ratio
        self.quadrature = GaussHermiteQuadrature1D()

    def to_one_hot(self, y, dtype):
        # convert a single label into a one-hot vector
        y_output_onehot = torch.zeros((y.shape[0], self.num_classes), dtype=dtype, device=y.device)
        return y_output_onehot.scatter_(1, y.unsqueeze(1), 1)

    def print_hyperparams(self):
        logging.info(f"output scale: "
                     f"{np.round_(detach_to_numpy(self.model.covar_module.outputscale.squeeze()), decimals=2)}")
        if self.kernel_func == "RBFKernel":
            lengthscale = detach_to_numpy(self.model.covar_module.base_kernel.lengthscale.squeeze())
            logging.info(f"length scale: "
                         f"{np.round_(lengthscale, decimals=2)}")
        elif self.kernel_func == "LinearKernel":
            variance = detach_to_numpy(self.model.covar_module.base_kernel.variance.squeeze())
            logging.info(f"variance: "
                         f"{np.round_(variance, decimals=2)}")

    def train_test_split(self, X, Y):

        # startified sampling
        unique_classes = torch.unique(Y, return_inverse=False, return_counts=False)
        for i, cls in enumerate(unique_classes):
            X_c, y_c = pytorch_take(X, Y, [cls.item()])
            k_fold = int(X_c.size(0) * self.predict_ratio)
            perm = torch.randperm(X_c.size(0))
            X_test = X_c[perm][:k_fold] if i == 0 else torch.cat((X_test, X_c[perm][:k_fold]), dim=0)
            Y_test = y_c[perm][:k_fold] if i == 0 else torch.cat((Y_test, y_c[perm][:k_fold]), dim=0)
            X_train = X_c[perm][k_fold:] if i == 0 else torch.cat((X_train, X_c[perm][k_fold:]), dim=0)
            Y_train = y_c[perm][k_fold:] if i == 0 else torch.cat((Y_train, y_c[perm][k_fold:]), dim=0)
        return X_train, X_test, Y_train, Y_test

    def forward_mll(self, X, Y, X_bar, to_print=True):

        X_bar = X_bar.reshape(X_bar.shape[0] * X_bar.shape[1], -1)

        self.num_steps = self.num_steps_train
        self.num_draws = self.num_draws_train

        model_state = self.fit(X, Y, X_bar)
        gibbs_state = self.gibbs_sample(model_state)

        # nmll with an average over the number of samples
        nmll = self.marginal_log_likelihood(gibbs_state.omega, model_state) / X.shape[0]

        if to_print:
            self.print_hyperparams()
            logging.info(f"Loss: {nmll.item() * X.shape[0]:.5f}, Avg. Loss: {nmll.item():.5f}")

        return nmll

    def forward_predictive(self, X, Y, X_bar, to_print=True):

        X_bar = X_bar.reshape(X_bar.shape[0] * X_bar.shape[1], -1)

        X_train, X_test, Y_train, Y_test = self.train_test_split(X, Y)

        self.num_steps = self.num_steps_train
        self.num_draws = self.num_draws_train

        model_state = self.fit(X_train, Y_train, X_bar)
        gibbs_state = self.gibbs_sample(model_state)

        dist = self.gaussian_posterior(gibbs_state.omega, model_state, X_test, X_bar)
        probs = torch.exp(self.quadrature(F.logsigmoid, dist)).mean(0)

        # to probabilities
        probs = probs.unsqueeze(1)
        preds = torch.cat((probs, 1 - probs), dim=1)
        loss = CE_loss(Y_test, preds, 2, reduction='mean')

        if to_print:
            self.print_hyperparams()
            logging.info(f"Loss: {loss.item() * X_test.shape[0]:.5f}, Avg. Loss: {loss.item():.5f}")

        return loss

    def predictive_posterior(self, X, Y, X_star, X_bar, is_first_iter=False):

        self.num_steps = self.num_steps_test
        self.num_draws = self.num_draws_test

        # at first iteration get omega
        if is_first_iter:
            model_state = self.fit(X, Y, X_bar)
            gibbs_state = self.gibbs_sample(model_state)
            self.save_state(model_state, gibbs_state)
        else:
            model_state = self.last_model_state
            gibbs_state = self.last_gibbs_state

        dist = self.gaussian_posterior(gibbs_state.omega, model_state, X_star, X_bar)
        probs = torch.exp(self.quadrature(F.logsigmoid, dist))
        probs_mean = probs.mean(0)

        return probs_mean

    def fit(self, X, Y, X_bar):

        N = X.shape[0]
        M = X_bar.shape[0]

        data = torch.cat((X_bar, X), dim=0)
        mu, K = self.model(data)
        mu = mu.unsqueeze(1).type(X.dtype)  # N x 1

        Kmm = K[:M, :M]
        Knm = K[:M, M:].t()
        Kmn = Knm.t()
        Knn = K[M:, M:]

        L = psd_safe_cholesky(Kmm)  # M x M
        Knm_Kmminv_kmn = Knm.matmul(torch.cholesky_solve(Kmn, L))
        Knm_Kmminv_mu = Knm.matmul(torch.cholesky_solve(mu[:M, :], L))
        Knn_Minus_Knm_Kmm_Kmn = torch.diag_embed(torch.diagonal(Knn - Knm_Kmminv_kmn))

        # stick break N vector: (ND * N) x C
        Y_one_hot = to_one_hot(Y, dtype=X.dtype)
        N_sb = N_vec(Y_one_hot).repeat(self.num_draws, 1)
        kappa = kappa_vec(Y_one_hot)

        return NodeModelState(
            Y=Y.clone(),
            M=M,
            N=N,
            N_sb=N_sb,
            kappa=kappa,
            mu=mu,
            K=K,
            L=L,
            Knm_Kmminv_kmn=Knm_Kmminv_kmn,
            Knm_Kmminv_mu=Knm_Kmminv_mu,
            Knn_Minus_Knm_Kmm_Kmn=Knn_Minus_Knm_Kmm_Kmn
        )

    def gibbs_sample(self, model_state):

        gibbs_state = self.initial_gibbs_state(model_state)

        # sample next state according to conditional posterior
        for _ in range(self.num_steps):
            gibbs_state = self.next_gibbs_state(model_state, gibbs_state)

        return gibbs_state

    def initial_gibbs_state(self, model_state):

        L = model_state.L
        M = model_state.M
        N = model_state.N
        ND = self.num_draws
        Knn_Minus_Knm_Kmm_Kmn = model_state.Knn_Minus_Knm_Kmm_Kmn
        Knm_Kmminv_mu = model_state.Knm_Kmminv_mu
        K = model_state.K
        Knm = K[M:, :M].unsqueeze(0)  # 1 x N x M

        # init u according to prior
        SN = torch.randn((M, ND), dtype=L.dtype, device=L.device)
        fbar_init = model_state.mu[:M, :] + L.matmul(SN)  # M x ND

        Knm_Kmminv_u = Knm.matmul(torch.cholesky_solve(fbar_init, L)).squeeze(0)

        # init f according to the prior conditional distribution
        SN = torch.randn((N, 1), dtype=L.dtype, device=L.device)
        mu_f_prior = model_state.mu[M:, :] + Knm_Kmminv_u - Knm_Kmminv_mu  # N x ND
        L_f_prior = torch.sqrt(Knn_Minus_Knm_Kmm_Kmn)
        f_init = mu_f_prior + L_f_prior.matmul(SN)  # N x ND

        # TODO: sample from actual PG prior
        omega_init = self.sample_omega(f_init, model_state)

        return NodeGibbsState(omega_init, f_init)

    def next_gibbs_state(self, model_state, gibbs_state):
        f_new = self.sample_f(gibbs_state.omega, model_state)
        omega_new = self.sample_omega(f_new, model_state)

        return NodeGibbsState(omega_new, f_new)

    # P(ω | Y, f)
    def sample_omega(self, f, model_state):
        """"
        Sample from polya-gamma distribution.
        :parm c - number of observations per sample
        :return flattened array of samples of size C * N * ND
        """
        N = model_state.N

        b = detach_to_numpy(model_state.N_sb).reshape(-1).astype(np.double)
        c = detach_to_numpy(f).reshape(-1).astype(np.double)
        ret = np.zeros_like(c)  # fill with samples

        self.ppg.pgdrawv(b, c, ret)

        omega = torch.tensor(ret, dtype=f.dtype, device=f.device).view(self.num_draws, N)  # [ND, N]
        return omega

    def sample_f(self, omega, model_state):
        dist = self.gaussian_posterior(omega, model_state)
        return dist.rsample()

    # P(f | Y, ω, X)
    def gaussian_posterior(self, omega, model_state, X_star=None, X_bar=None):
        M = model_state.M
        kappa = model_state.kappa  # N x 1
        Knn_Minus_Knm_Kmm_Kmn = model_state.Knn_Minus_Knm_Kmm_Kmn.unsqueeze(0)  # 1 x N x N
        Knm_Kmminv_mu = model_state.Knm_Kmminv_mu.unsqueeze(0)  # 1 x N x N
        Knm_Kmminv_kmn = model_state.Knm_Kmminv_kmn.unsqueeze(0)  # 1 x N x N
        L = model_state.L.unsqueeze(0)  # 1 x M x M

        K = model_state.K
        mu = model_state.mu

        MUn = mu[M:, :]

        Kmm = K[:M, :M].unsqueeze(0)  # 1 x M x M
        Knm = K[M:, :M].unsqueeze(0)  # 1 x N x M
        Kmn = K[:M, M:].unsqueeze(0)  # 1 x M x N
        Knn = K[M:, M:].unsqueeze(0)  # 1 x N x N

        if X_star is None:
            mu_s, Ks, Kss = MUn, Knm, Knn
            Ks_Kmminv_ks = Knm_Kmminv_kmn
            Ksm_Kmminv_mu = Knm_Kmminv_mu
        else:
            mu_star, K_star = self.model(torch.cat((X_bar, X_star), dim=0))
            mu_s = mu_star[M:].unsqueeze(1)  # N x 1
            Ks = K_star[M:, :M].unsqueeze(0)  # 1 x N x M
            Kss = K_star[M:, M:].unsqueeze(0)  # 1 x N x M

            Ks_Kmminv_ks = Ks.matmul(torch.cholesky_solve(Ks.permute(0, 2, 1), L))
            Ksm_Kmminv_mu = Ks.matmul(torch.cholesky_solve(mu[:M, :], L))

        Ω_inv = torch.diag_embed(1.0 / omega)  # ND x N x N

        Λ = torch.diagonal(Ω_inv + Knn_Minus_Knm_Kmm_Kmn, dim1=1, dim2=2)  # ND x N x N
        Λ_inv = torch.diag_embed(1.0 / Λ)

        Q = Kmm + Kmn.matmul(Λ_inv).matmul(Knm)  # ND x M x M
        L_Q = psd_safe_cholesky(Q)  # ND x M x M
        mu_f = (Ks.matmul(torch.cholesky_solve(Kmn, L_Q)).matmul(Λ_inv).matmul
                  (Ω_inv.matmul(kappa) - (MUn + Knm_Kmminv_mu))
                  + (mu_s - Ksm_Kmminv_mu)
                  ).squeeze(-1)

        Sigma_f = torch.diag_embed(torch.diagonal(
            Kss - Ks_Kmminv_ks + Ks.matmul(torch.cholesky_solve(Ks.permute(0, 2, 1), L_Q))
            , dim1=1, dim2=2)
        )
        L_Sigma_f = psd_safe_cholesky(Sigma_f)  # ND x M x M

        return MultivariateNormal(mu_f, scale_tril=L_Sigma_f)

    # ∑ (log(P(Y = c| ω, X)))
    def marginal_log_likelihood(self, omega, model_state):
        """
        Compute marginal likelihood with the given values of omega
        :param augmented_data:
        :return: log likelihood per class
        """
        kappa = model_state.kappa.t()  # 1 x N
        N = model_state.N
        M = model_state.M
        N_sb = model_state.N_sb  # (ND * N) x 1
        Knn_Minus_Knm_Kmm_Kmn = model_state.Knn_Minus_Knm_Kmm_Kmn.unsqueeze(0)  # 1 x N x N
        Knm_Kmminv_kmn = model_state.Knm_Kmminv_kmn.unsqueeze(0)  # 1 x N x N

        Ω_inv = torch.diag_embed(1.0 / omega)  # ND x N x N

        Λ = Ω_inv + Knn_Minus_Knm_Kmm_Kmn  # ND x N x N
        Sigma_Y = Λ + Knm_Kmminv_kmn  # ND x N x N

        mu_Y = model_state.mu[M:, :].t().repeat(self.num_draws, 1)  # 1 x N

        # The "observations" are the effective mean of the Gaussian likelihood given omega
        y_tilde = kappa / omega
        L_y = psd_safe_cholesky(Sigma_Y)
        p_y = MultivariateNormal(mu_Y, scale_tril=L_y)

        mll = p_y.log_prob(y_tilde) \
              + 0.5 * N * np.log(2 * np.pi) \
              - 0.5 * torch.log(omega).sum(-1) \
              + 0.5 * torch.sum((kappa.unsqueeze(1) ** 2) / omega, -1) \
              - torch.sum(N_sb.view(self.num_draws, N, -1) * np.log(2.0), dim=1).t()

        mll = - mll.mean()
        return mll

    def save_state(self, model_state, gibbs_state):

        N = model_state.N  # number of training samples
        M = model_state.M

        mu, K = model_state.mu.clone(), model_state.K.detach().clone()
        L = model_state.L.detach().clone()

        Knm_Kmminv_kmn = model_state.Knm_Kmminv_kmn.detach().clone()
        Knm_Kmminv_mu = model_state.Knm_Kmminv_mu.detach().clone()
        Knn_Minus_Knm_Kmm_Kmn = model_state.Knn_Minus_Knm_Kmm_Kmn.detach().clone()

        N_sb = model_state.N_sb
        kappa = model_state.kappa

        self.last_model_state = NodeModelState(
            Y=model_state.Y.clone(),
            M=M,
            N=N,
            N_sb=N_sb,
            kappa=kappa,
            mu=mu,
            K=K,
            L=L,
            Knm_Kmminv_kmn=Knm_Kmminv_kmn,
            Knm_Kmminv_mu=Knm_Kmminv_mu,
            Knn_Minus_Knm_Kmm_Kmn=Knn_Minus_Knm_Kmm_Kmn
        )
        self.last_gibbs_state = NodeGibbsState(gibbs_state.omega.detach().clone(),
                                             gibbs_state.f.detach().clone())