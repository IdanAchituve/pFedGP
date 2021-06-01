from gpytorch import kernels
from gpytorch import constraints
import gpytorch
import torch
from torch import nn
import torch.nn.functional as F

class GPModel(nn.Module):
    def __init__(self, jitter_val=1e-3):
        super().__init__()
        # mean and cov functions
        self.jitter_val = jitter_val

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        # L2 normalization
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)

        mean_x = self.mean_module(x2)
        covar_x = self.covar_module(x1, x2).add_jitter(jitter_val=self.jitter_val).evaluate()
        return mean_x, covar_x

    def _set_params(self, outputscale=8., lengthscale=1.):
        # init hyperparameters
        self.covar_module.outputscale = outputscale
        if self.kernel_function == 'LinearKernel':
            self.covar_module.base_kernel.variance = 1.
        elif self.kernel_function == 'RBFKernel':
            self.covar_module.base_kernel.lengthscale = lengthscale


class OneClassGPModel(GPModel):
    def __init__(self, kernel_function, jitter_val=1e-3):
        super(OneClassGPModel, self).__init__(jitter_val)

        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel_function = kernel_function
        if kernel_function == "RBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun = kernels.RBFKernel(lengthscale_constraint=constraints.GreaterThan(1e-2))
        elif kernel_function == "LinearKernel":
            self.ker_fun = kernels.LinearKernel()
        elif kernel_function == "MaternKernel":
            self.ker_fun = kernels.MaternKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.ker_fun)


class BatchedGPModel(GPModel):
    def __init__(self, kernel_function, jitter_val=1e-3, num_classes=200):
        super(BatchedGPModel, self).__init__(jitter_val)

        self.num_kernels = torch.Size([num_classes])
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=self.num_kernels)
        self.kernel_function = kernel_function
        if kernel_function == "RBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun = kernels.RBFKernel(batch_shape=self.num_kernels,
                                             lengthscale_constraint=constraints.GreaterThan(1e-2))
        elif kernel_function == "LinearKernel":
            self.ker_fun = kernels.LinearKernel(batch_shape=self.num_kernels)
        elif kernel_function == "MaternKernel":
            self.ker_fun = kernels.MaternKernel(batch_shape=self.num_kernels)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.ker_fun)
        self.kernel_function = kernel_function