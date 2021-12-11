from collections import namedtuple
import pypolyagamma
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from torch import nn
from pFedGP.kernel_class import OneClassGPModel
from scipy.special import logsumexp
import torch.nn.functional as F

from utils import *

NodeGibbsState = namedtuple("NodeGibbsState", ["omega", "f"])
NodeModelState = namedtuple(
    "NodeModelState",
    ["N", "N_sb", "mu", "K", "L", "Kinv", "Kinv_mu", "X", "Y", "C", "kappa"],
)


class pFedGPFull(nn.Module):
    def __init__(self,
                 kernel_func,
                 num_classes=2,
                 num_steps=10,
                 num_draws=20,
                 num_steps_test=10,
                 num_draws_test=30,
                 predict_ratio=0.4):

        super(pFedGPFull, self).__init__()
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

    def forward_mll(self, X, Y, to_print=True):

        self.num_steps = self.num_steps_train
        self.num_draws = self.num_draws_train

        model_state = self.fit(X, Y)
        gibbs_state = self.gibbs_sample(model_state)

        # nmll with an average over the number of samples
        nmll = self.marginal_log_likelihood(gibbs_state.omega, model_state) / X.shape[0]

        if to_print:
            self.print_hyperparams()
            logging.info(f"Loss: {nmll.item() * X.shape[0]:.5f}, Avg. Loss: {nmll.item():.5f}")

        return nmll

    def forward_predictive(self, X, Y, to_print=True):

        X_train, X_test, Y_train, Y_test = self.train_test_split(X, Y)

        self.num_steps = self.num_steps_train
        self.num_draws = self.num_draws_train

        model_state = self.fit(X_train, Y_train)
        gibbs_state = self.gibbs_sample(model_state)

        dist = self.predictive_dist(model_state, gibbs_state, X_test, X_train)
        probs = torch.exp(self.quadrature(F.logsigmoid, dist)).mean(0)

        # to probabilities
        probs = probs.unsqueeze(1)
        preds = torch.cat((probs, 1 - probs), dim=1)
        loss = CE_loss(Y_test, preds, 2, reduction='mean')

        if to_print:
            self.print_hyperparams()
            logging.info(f"Loss: {loss.item() * X_test.shape[0]:.5f}, Avg. Loss: {loss.item():.5f}")

        return loss

    def predictive_posterior(self, X, Y, X_star, is_first_iter=False):
        self.num_steps = self.num_steps_test
        self.num_draws = self.num_draws_test

        # at first iteration get omega
        if is_first_iter:
            model_state = self.fit(X, Y)
            gibbs_state = self.gibbs_sample(model_state)
            self.save_state(model_state, gibbs_state)
        else:
            model_state = self.last_model_state
            gibbs_state = self.last_gibbs_state

        dist = self.predictive_dist(model_state, gibbs_state, X_star, X)
        probs = torch.exp(self.quadrature(F.logsigmoid, dist))
        probs_mean = probs.mean(0)

        return probs_mean

    def fit(self, X, Y):
        C = 1
        N = X.shape[0]

        mu, K = self.model(X)
        mu = mu.unsqueeze(1).type(X.dtype)  # N x 1

        L = psd_safe_cholesky(K)
        Kinv = torch.cholesky_solve(torch.eye(K.shape[0], dtype=K.dtype, device=K.device), L)
        Kinv_mu = Kinv.matmul(mu)

        L = L.unsqueeze(0)
        K = K.unsqueeze(0)
        Kinv = Kinv.unsqueeze(0)
        Kinv_mu = Kinv_mu.unsqueeze(0)

        # stick break N vector: (ND * N) x C
        Y_one_hot = to_one_hot(Y, dtype=X.dtype)
        N_sb = N_vec(Y_one_hot).repeat(self.num_draws, 1)
        kappa = kappa_vec(Y_one_hot)

        return NodeModelState(
            N=N,
            N_sb=N_sb,
            mu=mu,
            K=K,
            L=L,
            Kinv_mu=Kinv_mu,
            Kinv=Kinv,
            X=X.clone().detach(),
            Y=Y.clone(),
            C=C,
            kappa=kappa
        )

    def gibbs_sample(self, model_state):

        gibbs_state = self.initial_gibbs_state(model_state)

        # sample next state according to conditional posterior
        for _ in range(self.num_steps):
            gibbs_state = self.next_gibbs_state(model_state, gibbs_state)

        return gibbs_state

    def initial_gibbs_state(self, model_state):

        L = model_state.L
        N = model_state.N
        ND = self.num_draws

        # init to the prior mean
        SN = torch.normal(mean=torch.zeros(1 * ND * N, dtype=L.dtype, device=L.device),
                          std=torch.ones(1 * ND * N, dtype=L.dtype, device=L.device)).view(ND, N, 1)
        f_init = model_state.mu.unsqueeze(0) + L.matmul(SN)
        f_init = f_init.squeeze(-1)

        # TODO: sample from actual PG prior
        omega_init = self.sample_omega(f_init, model_state)

        return NodeGibbsState(omega_init, f_init)

    def next_gibbs_state(self, model_state, gibbs_state):
        f_new = self.gaussian_conditional(gibbs_state.omega, model_state)
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

    # P(f | Y, ω, X)
    def gaussian_conditional(self, omega, model_state):
        kappa = model_state.kappa.t()
        #Ω = torch.diag_embed(omega)
        omega = omega.clamp(min=1e-32)

        # Set the precision for invalid points to zero
        Kinv_mu = model_state.Kinv_mu
        Kinv = model_state.Kinv

        #sigma_tilde = torch.inverse(Kinv + Ω)
        Ω_inv = torch.diag_embed(1.0 / omega)
        L_noisy = psd_safe_cholesky(model_state.K + Ω_inv)
        sigma_tilde = model_state.K - model_state.K.matmul(
            torch.cholesky_solve(model_state.K, L_noisy)
        )

        # upper triangular of covariance matrices, each corresponds to different combination
        # of class and draw
        mu_tilde = sigma_tilde.matmul(kappa.unsqueeze(-1) + Kinv_mu).squeeze(-1)

        L_tilde = psd_safe_cholesky(sigma_tilde)
        fs = torch.distributions.MultivariateNormal(mu_tilde, scale_tril=L_tilde).rsample()
        return fs

    # ∑ (log(P(Y = c| ω, X)))
    def marginal_log_likelihood(self, omega, model_state):
        """
        Compute marginal likelihood with the given values of omega
        :param augmented_data:
        :return: log likelihood per class
        """
        kappa = model_state.kappa.t()  # 1 x N
        N = model_state.N
        N_sb = model_state.N_sb  # (ND * N) x 1
        K = model_state.K

        # prevents division by 0
        omega = omega.clamp(min=1e-16)

        # diagonal matrix for each combination of class, and number of draws [ND, N, N]
        Ω_inv = torch.diag_embed(1.0 / omega)
        z_Sigma = K + Ω_inv
        z_mu = model_state.mu.t()  # 1 x N

        # The "observations" are the effective mean of the Gaussian likelihood given omega
        # when omega is zero kappa should be zero as well
        z = kappa / omega
        L_z = psd_safe_cholesky(z_Sigma)
        p_y = torch.distributions.MultivariateNormal(z_mu, scale_tril=L_z)

        mll = p_y.log_prob(z) \
              + 0.5 * N * np.log(2 * np.pi) \
              - 0.5 * torch.log(omega).sum(-1) \
              + 0.5 * torch.sum((kappa.unsqueeze(1) ** 2) / omega, -1) \
              - torch.sum(N_sb.view(self.num_draws, N, -1) * np.log(2.0), dim=1).t()

        mll = - mll.mean()
        return mll

    # P(f^* | ω, Y, x^*, X)) & P(y^* | f^*)
    def predictive_dist(self, model_state, gibbs_state, X_star, X):

        omega = gibbs_state.omega  # ND x N
        kappa = model_state.kappa.t()  # 1 x N
        z_mu = model_state.mu.t()  # 1 x N
        N = model_state.N

        Xon = torch.cat((X, X_star), dim=0)
        mu_on, K_on = self.model(Xon)

        # recompute kernel since it might have changed due to backward operation 1 x N x N
        K = K_on[:N, :N].unsqueeze(0)
        # covariance function between existing and new samples 1 x N x M
        K_s = K_on[:N, N:].unsqueeze(0)
        # covariance function between new samples 1 x M x M
        K_ss = K_on[N:, N:].unsqueeze(0)

        mu_s = mu_on[N:].unsqueeze(1).type(X_star.dtype)  # M x 1

        omega = omega.clamp(min=1e-16)
        z = kappa / omega - z_mu  # N x 1

        # diagonal matrix for each draw ND x N x N
        Ω_inv = torch.diag_embed(1.0 / omega)
        L_noisy = psd_safe_cholesky(K + Ω_inv)

        # mu^* + K^* x ((K + Ω^-1)^-1 x (Ω^-1 x kappa - mu))
        mu_pred = mu_s + K_s.permute(0, 2, 1).matmul(
            torch.cholesky_solve(z.unsqueeze(-1), L_noisy))
        mu_pred = mu_pred.squeeze(-1)  # ND x M

        Sigma_pred = torch.diagonal(K_ss - K_s.permute(0, 2, 1).
                                    matmul(torch.cholesky_solve(K_s, L_noisy)), dim1=1, dim2=2)
        Sigma_pred = torch.diag_embed(Sigma_pred)  # [ND, M, M]
        L_s = psd_safe_cholesky(Sigma_pred)
        dist = torch.distributions.MultivariateNormal(mu_pred, scale_tril=L_s)
        return dist

    def save_state(self, model_state, gibbs_state):

        C = model_state.C
        N = model_state.N

        # mean function - initialized to zero or empirical value for all classes [N x (C-1)]
        mu = model_state.mu.detach().clone()

        # covariance function C x 1 x N x N
        K = model_state.K.detach().clone()
        # stick break N vector: (ND * N) x C
        N_sb = model_state.N_sb.detach().clone()

        L = model_state.L.detach().clone()
        Kinv = model_state.Kinv.detach().clone()
        Kinv_mu = model_state.Kinv_mu.detach().clone()
        kappa = model_state.kappa.detach().clone()

        self.last_model_state = NodeModelState(
                                    N=N,
                                    N_sb=N_sb,
                                    mu=mu,
                                    K=K,
                                    L=L,
                                    Kinv=Kinv,
                                    Kinv_mu=Kinv_mu,
                                    X=model_state.X.clone(),
                                    Y=model_state.Y.clone(),
                                    C=C,
                                    kappa=kappa
                                )
        self.last_gibbs_state = NodeGibbsState(gibbs_state.omega.detach().clone(),
                                             gibbs_state.f.detach().clone())


class pFedGPIPData(pFedGPFull):

    def __init__(self,
                 kernel_func,
                 num_classes=2,
                 num_steps=10,
                 num_draws=20,
                 num_steps_test=10,
                 num_draws_test=30,
                 predict_ratio=0.4,
                 balance_classes=False):

        super(pFedGPIPData, self).__init__(kernel_func, num_classes, num_steps,
                                           num_draws, num_steps_test, num_draws_test, predict_ratio)
        self.balance_classes = balance_classes

    def forward_predicitive(self, X, Y, X_bar=None, Y_bar=None, to_print=True, *args, **kwargs):
        X_bar = X_bar.reshape(X_bar.shape[0] * X_bar.shape[1], -1)

        self.num_steps = self.num_steps_train
        self.num_draws = self.num_draws_train

        model_state = self.fit(X_bar, Y_bar)
        gibbs_state = self.gibbs_sample(model_state)

        self.save_state(model_state, gibbs_state)

        dist = self.predictive_dist(model_state, gibbs_state, X, X_bar)
        probs = torch.exp(self.quadrature(F.logsigmoid, dist)).mean(0)

        # to probabilities
        probs = probs.unsqueeze(1)
        preds = torch.cat((probs, 1 - probs), dim=1)
        if self.balance_classes:
            node_labels, node_label_counts = torch.unique(Y, return_counts=True)
            # P(Y_* | X_*) = (P(Y)/Q(Y)) * Q(Y_*|X_*); where P(Y) is unbalanced and Q(Y) is balanced
            preds[:, 0] *= node_label_counts[0]
            preds[:, 1] *= node_label_counts[1]
            # normalize
            preds /= torch.sum(preds, dim=1, keepdim=True)

        loss = CE_loss(Y, preds, 2, reduction='mean')

        if to_print:
            self.print_hyperparams()
            logging.info(f"Loss: {loss.item() * X.shape[0]:.5f}, Avg. Loss: {loss.item():.5f}")

        return loss

    def predictive_posterior(self, X_star, X=None, Y=None, is_first_iter=False):
        self.num_steps = self.num_steps_test
        self.num_draws = self.num_draws_test

        # at first iteration get omega
        if is_first_iter:
            model_state = self.fit(X, Y)
            gibbs_state = self.gibbs_sample(model_state)
            self.save_state(model_state, gibbs_state)
        else:
            model_state = self.last_model_state
            gibbs_state = self.last_gibbs_state

        # recalc kernel since the last X might have changed due to backprop operation
        dist = self.predictive_dist(model_state, gibbs_state, X_star, X)
        probs = torch.exp(self.quadrature(F.logsigmoid, dist))
        probs_mean = probs.mean(0)

        return probs_mean


class pFedGPFullBound(pFedGPFull):

    def __init__(self,
                 kernel_func,
                 num_classes=2,
                 num_steps=2000,
                 num_draws=5,
                 num_steps_test=2000,
                 num_draws_test=5,
                 predict_ratio=0.5,
                 start_collect=100):

        super(pFedGPFullBound, self).__init__(kernel_func, num_classes, num_steps, num_draws, num_steps_test,
                                               num_draws_test, predict_ratio)
        self.start_collect = start_collect
        self.delta = 0.01

    def kl_ber(self, q, t):
        return q * np.log(q / (q + t) + 1e-12) + (1 - q) * np.log((1 - q)/(1 - (q + t)) + 1e-12)

    def Dinv(self, a, b, q, epsilon, tol=1e-6):
        """
        approximates a root, R, of f bounded
        by a and b to within tolerance
        | f(m) | < tol with t the midpoint
        between a and b Recursive implementation
        """

        # get midpoint
        t = (a + b) / 2
        diff = self.kl_ber(q, t) - epsilon
        while np.abs(diff) > tol:

            if diff > 0:
                a = a
                b = t
            elif diff < 0:
                a = t
                b = b
            else:
                raise print("error encountered")

            t = (a + b) / 2  # new midpoint
            diff = self.kl_ber(q, t) - epsilon  # new diff

        return t

    def forward_get_risk(self, X_star, Y_star, X=None, Y=None):
        self.num_steps = self.num_steps_test
        self.num_draws = self.num_draws_test
        N = X.shape[0]

        # at first iteration get omega
        model_state = self.fit(X, Y)
        gibbs_state, samples, f_dist = self.gibbs_sample(model_state)

        # empirical risk
        dist = self.predictive_dist(model_state, gibbs_state, X, X)
        #dist = f_dist  # Q(f | X, Y, ω)
        # sample 1000 functions from the posterior: f ~ Q(f | X, S, ω)
        train_preds = dist.rsample(torch.tensor([300]))  # [#samples, ND, N]
        Y_train = 1 - Y.unsqueeze(0).unsqueeze(0)  # the model considers 0's as 1's and vice versa
        preds_sign = torch.sign(train_preds)
        preds_sign[preds_sign == -1] = 0
        empirical_gibbs_error = torch.mean(torch.abs(Y_train - preds_sign))  # average over ω's, functions and samples
        empirical_gibbs_error = detach_to_numpy(empirical_gibbs_error)

        mu_fs = dist.mean.mean(0)  # average over ω's the mean function
        Y_train = 1 - Y  # the model considers 0's as 1's and vice versa
        preds_sign = torch.sign(mu_fs)
        preds_sign[preds_sign == -1] = 0
        empirical_bayes_error = torch.mean(torch.abs(Y_train - preds_sign))  # average over samples
        empirical_bayes_error = detach_to_numpy(empirical_bayes_error)

        # ood_generalization
        dist = self.predictive_dist(model_state, gibbs_state, X_star, X)
        test_preds = dist.rsample(torch.tensor([300]))  # [#samples, ND, N]
        Y_star_gibbs = 1 - Y_star.unsqueeze(0).unsqueeze(0)  # the model considers 0's as 1's and vice versa
        preds_sign = torch.sign(test_preds)
        preds_sign[preds_sign == -1] = 0
        gene_gibbs_error = torch.mean(torch.abs(Y_star_gibbs - preds_sign))  # average over ω's, functions and samples
        gene_gibbs_error = detach_to_numpy(gene_gibbs_error)

        mu_fs = dist.mean.mean(0)  # average over ω's the mean function
        Y_star_bayes = 1 - Y_star  # the model considers 0's as 1's and vice versa
        preds_sign = torch.sign(mu_fs)
        preds_sign[preds_sign == -1] = 0
        gene_bayes_error = torch.mean(torch.abs(Y_star_bayes - preds_sign))  # average over samples
        gene_bayes_error = detach_to_numpy(gene_bayes_error)

        # calc kl
        e_kl_term = self.expected_kl(samples, model_state)
        neg_MI = self.MI(samples, model_state)
        kl_qf_pf = detach_to_numpy(e_kl_term + neg_MI)

        epsilon = (1 / N) * (kl_qf_pf + np.log((N + 1) / self.delta))
        Rs_gibbs = empirical_gibbs_error + self.Dinv(a=0, b=1-empirical_gibbs_error,
                                                     q=empirical_gibbs_error, epsilon=epsilon)
        Rs_bayes = empirical_bayes_error + self.Dinv(a=0, b=1-empirical_bayes_error,
                                                     q=empirical_bayes_error, epsilon=epsilon)

        Bs_gibbs = empirical_gibbs_error + np.sqrt((kl_qf_pf + np.log((2 * np.sqrt(N)) / self.delta)) / (2 * N))
        Bs_bayes = empirical_bayes_error + np.sqrt((kl_qf_pf + np.log((2 * np.sqrt(N)) / self.delta)) / (2 * N))

        probs = torch.exp(self.quadrature(F.logsigmoid, dist))
        probs_mean = probs.mean(0)

        return (probs_mean, Rs_gibbs, Rs_bayes, Bs_gibbs, Bs_bayes, gene_gibbs_error, gene_bayes_error,
                empirical_gibbs_error, empirical_bayes_error, e_kl_term, neg_MI)

    def gibbs_sample(self, model_state):

        samples = []
        gibbs_state = self.initial_gibbs_state(model_state)
        collect_every = 3

        # sample next state according to conditional posterior
        for i in range(self.num_steps):
            gibbs_state, log_post_omega, f_dist = self.next_gibbs_state(model_state, gibbs_state)

            if i >= self.start_collect and i % collect_every == 0:
                for chain in range(self.num_draws):
                    samples.append((NodeGibbsState(gibbs_state.omega[chain:chain+1, ...].detach().clone(),
                                                 gibbs_state.f[chain:chain+1, ...].detach().clone()),
                                    log_post_omega[chain]))

        return gibbs_state, samples, f_dist

    def next_gibbs_state(self, model_state, gibbs_state):
        f_dist = self.gaussian_conditional(gibbs_state.omega, model_state)
        f_new = f_dist.rsample()
        omega_new, log_post_omega = self.sample_omega(f_new, model_state)

        return NodeGibbsState(omega_new, f_new), log_post_omega, f_dist

    def initial_gibbs_state(self, model_state):

        L = model_state.L
        N = model_state.N
        ND = self.num_draws

        # init to the prior mean
        SN = torch.normal(mean=torch.zeros(1 * ND * N, dtype=L.dtype, device=L.device),
                          std=torch.ones(1 * ND * N, dtype=L.dtype, device=L.device)).view(ND, N, 1)
        f_init = model_state.mu.unsqueeze(0) + L.matmul(SN)
        f_init = f_init.squeeze(-1)

        # TODO: sample from actual PG prior
        omega_init, _ = self.sample_omega(f_init, model_state)

        return NodeGibbsState(omega_init, f_init)

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
        log_post_omega = np.sum(np.log(pypolyagamma.pgpdf(ret, b, c)).reshape(self.num_draws, N), axis=1)

        omega = torch.tensor(ret, dtype=f.dtype, device=f.device).view(self.num_draws, N)  # [ND, N]

        return omega, log_post_omega

    # P(f | Y, ω, X)
    def gaussian_conditional(self, omega, model_state):
        kappa = model_state.kappa.t()
        # Ω = torch.diag_embed(omega)
        omega = omega.clamp(min=1e-32)

        # Set the precision for invalid points to zero
        Kinv_mu = model_state.Kinv_mu
        Kinv = model_state.Kinv

        # sigma_tilde = torch.inverse(Kinv + Ω)
        Ω_inv = torch.diag_embed(1.0 / omega)
        L_noisy = psd_safe_cholesky(model_state.K + Ω_inv)
        sigma_tilde = model_state.K - model_state.K.matmul(
            torch.cholesky_solve(model_state.K, L_noisy)
        )

        # upper triangular of covariance matrices, each corresponds to different combination
        # of class and draw
        mu_tilde = sigma_tilde.matmul(kappa.unsqueeze(-1) + Kinv_mu).squeeze(-1)

        L_tilde = psd_safe_cholesky(sigma_tilde)
        fs = torch.distributions.MultivariateNormal(mu_tilde, scale_tril=L_tilde)
        return fs

    # E_q(ω) [KL(Q(f| ω, X_bar, Y_bar) || P(f | X))]
    def expected_kl(self, samples, model_state):

        K = model_state.K
        Kinv = model_state.Kinv
        Kinv_mu = model_state.Kinv_mu
        kappa = model_state.kappa.t()
        N = model_state.N
        mu = model_state.mu
        M = len(samples)

        log_det_K = torch.logdet(K)

        e_kl_term = 0
        for sample in samples:
            omega = sample[0].omega.clamp(min=1e-32)

            # Cov of the posterior (Kinv + Ω)^-1
            Ω_inv = torch.diag_embed(1.0 / omega)
            L_noisy = psd_safe_cholesky(model_state.K + Ω_inv)
            Σ = model_state.K - model_state.K.matmul(
                torch.cholesky_solve(model_state.K, L_noisy)
            )
            log_det_Σ = torch.logdet(Σ)

            termA = log_det_K - log_det_Σ
            termB = - N
            termC = torch.sum(torch.diagonal(Kinv.matmul(Σ), dim1=1, dim2=2), dim=1)

            mu_tilde = (Σ.matmul(kappa.unsqueeze(-1) + Kinv_mu).squeeze(-1) - mu.t()).unsqueeze(1)
            termD = mu_tilde.matmul(Kinv.matmul(mu_tilde.permute(0, 2, 1))).squeeze(-1).squeeze(-1)
            # avg. kl over chains per sample
            e_kl_term += (1/2) * torch.mean(termA + termB + termC + termD)

        return (1 / M) * e_kl_term

    def MI(self, samples, model_state):

        M = len(samples) - 1  #  don't take the corresponding f of omega
        N = model_state.N

        log_posterior_omega = 0
        log_q_omega = []

        b = detach_to_numpy(model_state.N_sb.reshape((self.num_draws, -1))[0, :])
        b = np.repeat(b, repeats=M).astype(np.double)

        # prepare all fs
        fs = []
        for s in samples:
            fs.append(detach_to_numpy(s[0].f))

        fs = np.concatenate(fs, axis=0).astype(np.double)

        for k, sample_k in enumerate(samples):
            omega_k = detach_to_numpy(sample_k[0].omega.clamp(min=1e-32))
            log_posterior_omega += sample_k[1]

            f_not_k = np.concatenate((fs[:k, :], fs[k+1:, :]), axis=0).reshape(-1)
            dup_omega_k = np.repeat(omega_k, repeats=M, axis=0).reshape(-1).astype(np.double)

            log_q_omega.append(np.sum(np.log(pypolyagamma.pgpdf(dup_omega_k, b, f_not_k).reshape(M, -1)), axis=1))

            if k % 50 == 0:
                logging.info(f"Finished Estimating ω_{k}")

        # log q(ω_k | X, Y) ~ log (1 / M) * sum_over_f exp(q(ω_k | f, X, Y))
        log_q_omega_k = logsumexp(np.asarray(log_q_omega), axis=1) - np.log(M)
        # (1 / M) * sum_over_ω q(ω | X, Y)
        avg_log_q_omega = np.mean(log_q_omega_k)
        return avg_log_q_omega - (log_posterior_omega / (M + 1))