# define the actual GP model (kernels, inducing points, etc.)
# https://notebook.community/jrg365/gpytorch/examples/02_Simple_GP_Classification/Simple_GP_Classification
import gpytorch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, DeltaVariationalDistribution, \
    NaturalVariationalDistribution


class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = DeltaVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.PeriodicKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)