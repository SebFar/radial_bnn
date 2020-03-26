import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from radial_layers import distributions

class SVI_Base(nn.Module):
    """
    Base class for Stochastic Variational Inference. Called by layers. Currently implemented:
    SVI_Linear
    SVI_Conv2D
    SVIMaxPool2D
    SVIGlobalMaxPool2D
    SVIAverageMaxPool2D
    """

    def __init__(self,
                 weight_shape,
                 bias_shape,
                 variational_distribution,
                 prior,
                 use_bias):
        super(SVI_Base, self).__init__()

        self.data_type = torch.float32

        self.weight_rhos = nn.Parameter(torch.empty(weight_shape, dtype=self.data_type))
        self.weight_mus = nn.Parameter(torch.empty(weight_shape, dtype=self.data_type))
        self.weight = Variable(torch.empty(weight_shape, dtype=self.data_type))
        self.use_bias = use_bias
        if use_bias:
            self.bias_rhos = nn.Parameter(torch.empty(bias_shape, dtype=self.data_type))
            self.bias_mus = nn.Parameter(torch.empty(bias_shape, dtype=self.data_type))
            self.bias = Variable(torch.empty(bias_shape, dtype=self.data_type))
        else:
            self.register_parameter('bias_rhos', None)
            self.register_parameter('bias_mus', None)
            self.register_parameter('bias', None)

        # The prior log probability density function is any function that takes takes a Tensor of weight and returns
        # a Tensor of the same shape with the log probability density of those points.
        # gaussian_prior is implemented
        assert hasattr(distributions, prior['name']), "The prior named in config is not defined in utils.distributions"
        prior_args = copy.deepcopy(prior)
        prior_args["log2pi"] = torch.log(Variable(torch.from_numpy(np.array(2.0 * np.pi)).type(self.data_type),
                                         requires_grad=False))
        prior_args["device"] = 'cpu'
        if torch.cuda.is_available():
            prior_args["log2pi"] = prior_args["log2pi"].cuda()
            prior_args["device"] = 'cuda'
        self.prior_log_pdf = getattr(distributions, prior_args['name'])(**prior_args)

        # The variational distribution must take a size and return a sample from the noise distribution of the same size
        assert hasattr(distributions, variational_distribution), "The variational distribution is not defined in util.distributions"
        self.noise_distribution = getattr(distributions, variational_distribution)

        # The pretraining flag is controlled in the training loop
        # In the configuration file, set trainer["pretrain_epochs"] to a non-zero integer
        # While pretraining, no noise is sampled and only the means are optimized with an NLL loss
        # This helps stabilize training and is especially important for standard
        # MFVI with multivariate Gaussians.
        self.pretraining = False


    def _rho_to_sigma(self, rho):
        """
        We actually parameterize sigma with rho.
         Sigma is softplus rho, which ensures that we have positive standard deviation.
        :param rho: tensor of rhos
        :return: tensor of sigmas
        """
        return torch.log(1 + torch.exp(rho))

    def entropy(self):
        """
        Calculates the entropy:
        -\int q(w) log q(w)
        of the variational posterior up to a constant.
        For both the radial and multivariate Gaussian approximating distributions, this is:
        \sum_i log \sigma_i + c
        where i indexes over the weights.
        Returns: entropy of the approximate posterior up to a constant.
        """
        if not self.pretraining:
            entropy = torch.sum(torch.log(self._rho_to_sigma(self.weight_rhos)))
            if self.use_bias:
                entropy += torch.sum(torch.log(self._rho_to_sigma(self.bias_rhos)))
            return entropy
        else:
            return 0

    def cross_entropy(self):
        """
        Estimates the cross entropy between the variational posterior and prior
        - \int q(w) log(p(w)) dw
        using Monte Carlo integration.
        We find that this is a fairly low-variance estimator.
        Returns: cross-entropy

        """
        if not self.pretraining:
            weight_log_prior_mean_over_epsilon = torch.mean(self.prior_log_pdf(self.weight), dim=0)
            cross_entropy = -torch.sum(weight_log_prior_mean_over_epsilon)
            if self.use_bias:
                bias_log_prior_mean_over_epsilon = torch.mean(self.prior_log_pdf(self.bias), dim=0)
                cross_entropy -= torch.sum(bias_log_prior_mean_over_epsilon)
            return cross_entropy
        else:
            return 0

    def is_pretraining(self, pretraining_on):
        self.pretraining = pretraining_on
        return 1


class SVI_Linear(SVI_Base):
    """Models an independent Gaussian/mean-field approximation neural network. Based on
    pytorch module for nn.Linear"""

    def __init__(self,
                 in_features,
                 out_features,
                 initial_rho,
                 initial_mu,
                 variational_distribution,
                 prior,
                 use_bias=True):
        """
        Initializes weights and biases of a linear layer with stochastic variational inference over the weights.
        :param in_features: Number of inputs features to the layer
        :param out_features: Number of outputs from the leayer
        :param initial_rho: controls starting variance of layer (sigma = log(1+exp(rho))
        :param initial_mu: initial variance of mu as a zero-mean Gaussian or "he" uses Kaiming He initialization
        :param use_bias: flag for use of bias term default True
        """
        super(SVI_Linear, self).__init__((out_features, in_features),
                                         (out_features),
                                         variational_distribution,
                                         prior,
                                         use_bias)
        self.reset_parameters(initial_rho, initial_mu)


    def reset_parameters(self, initial_rho, mu_std):
        """Randomly populates mus by Gaussian distribution around zero
        and sets all rhos to preset value"""
        if mu_std == 'he':
            # He Kaiming for mus, assuming Leaky ReLUs with gradient -0.2
            fan_in = self.weight_mus.shape[1]
            std = math.sqrt(1.92 / fan_in)
        elif isinstance(mu_std, (int, float)):
            std = mu_std
        else:
            ValueError("Standard deviation of mu was {}. Expected 'he' or an int/float")
        self.weight_rhos.data.normal_(initial_rho, std=0.5)
        self.weight_mus.data.normal_(std=std)
        if self.bias_mus is not None:
            self.bias_rhos.data.normal_(initial_rho, std=0.5)
            self.bias_mus.data.normal_(std=std)

    def forward(self, x):
        """
        Computes the weights using reparameterization trick and then does a forward pass
        :param x: tensor of examples [examples, samples, features]
        :return: tensor of features to next layer
        """
        if not self.pretraining:
            # We transform our parameterisation in rho into sigma
            weight_sigma = self._rho_to_sigma(self.weight_rhos)  # [in_features, out_features]
            if self.use_bias:
                bias_sigma = self._rho_to_sigma(self.bias_rhos)  # [out_features]
            # Now we compute the random noise
            # We deduce the number of training samples from the second dimension of the input data
            train_samples = x.size()[1]
            # torch.Size() has base class tuple, so we add a singleton for the new size
            # This gives weight_epsilon size (training_samples, in_features, out_features)
            weight_epsilon = Variable(self.noise_distribution((train_samples,) + self.weight_mus.size()))
            if self.use_bias:
                # bias_epsilon [training_samples, out_features]
                bias_epsilon = Variable(self.noise_distribution((train_samples,) + self.bias_mus.size()))
            # And determine the parameters *w*

            self.weight = torch.addcmul(self.weight_mus, weight_sigma, weight_epsilon)
            output = torch.einsum('ijk,jlk->ijl', [x, self.weight])
            if self.use_bias:
                self.bias = torch.addcmul(self.bias_mus, bias_sigma, bias_epsilon)
                output = output + self.bias
        else:
            output = torch.einsum('ijk,lk->ijl', [x, self.weight_mus])
            if self.use_bias:
                output = output + self.bias_mus
        return output


class _SVIConvNd(SVI_Base):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 transposed,
                 output_padding,
                 groups,
                 use_bias,
                 variational_distribution,
                 prior,
                 initial_rho,
                 initial_mu_std):

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if transposed:
            weight_shape = (in_channels, out_channels // groups, *kernel_size)
        else:
            weight_shape = (out_channels, in_channels // groups, *kernel_size)
        bias_shape = (out_channels)
        super(_SVIConvNd, self).__init__(weight_shape, bias_shape, variational_distribution, prior, use_bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        self.reset_parameters(initial_rho, initial_mu_std)

    def reset_parameters(self, initial_rho, mu_std):
        self.weight_rhos.data.normal_(initial_rho, std=0.5)
        if mu_std == 'he':
            # Using pytorch's recommendation for Leaky Relu :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
            torch.nn.init.kaiming_uniform_(self.weight_mus, math.sqrt(1.92))
        elif isinstance(mu_std, (int, float)):
            self.weight_mus.data.normal_(std=mu_std)
        else:
            ValueError("Standard deviation of mu was {}. Expected 'he' or an int/float")
        if self.bias_mus is not None:
            self.bias_rhos.data.normal_(initial_rho, std=0.5)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_mus)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias_mus, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class SVIConv2D(_SVIConvNd):
    """Models an independent Gaussian/mean-field approximation neural network. Based on
    pytorch module for nnConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, variational_distribution, prior, initial_rho, mu_std,
                 stride=(1,1), padding=0, dilation=1, groups=1, bias=True):
        if dilation != 1:
            raise NotImplementedError
        if groups != 1:
            raise NotImplementedError
        if padding < 0:
            raise ValueError("Padding for SVIConv2D must be 0 or greater.")
        if stride[0] < 1 or stride[1] < 1:
            raise ValueError("Padding for SVIConv2D must be 1 or greater.")
        super(SVIConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, 0, groups, bias, variational_distribution, prior, initial_rho, mu_std)

    def forward(self, x):
        """
        Computes the weights using reparameterization trick and then does a forwards pass
        :param x: tensor of examples [examples, samples, in_channels, H, W]
        :return: tensor of features to next layer
        """
        if not self.pretraining:
            # We transform our parameterisation in rho into sigma
            weight_sigma = self._rho_to_sigma(self.weight_rhos)  # [out_channels, in_channels, H, W]
            if self.use_bias:
                bias_sigma = self._rho_to_sigma(self.bias_rhos)  # [out_channels]

            # We deduce the number of variational training samples from the second dimension of the input data
            train_samples = x.shape[1]

            # torch.Size() has base class tuple, so we add a singleton for the new size
            # This gives weight_epsilon size (training_samples, out, in, *kernel_size)
            weight_epsilon = Variable(self.noise_distribution((train_samples,) + self.weight_mus.size()))
            self.weight = torch.addcmul(self.weight_mus, weight_sigma, weight_epsilon)
            if self.use_bias:
                # bias_epsilon [training_samples, out_channels]
                bias_epsilon = Variable(self.noise_distribution((train_samples,) + self.bias_mus.size()))
                self.bias = torch.addcmul(self.bias_mus, bias_sigma, bias_epsilon)  # [samples, out_channels]
        else:
            self.weight = self.weight_mus.unsqueeze(0)
            if self.use_bias:
                self.bias = self.bias_mus.unsqueeze(0)
        # Add padding
        if self.padding != 0:
            x = torch.nn.functional.pad(x, [self.padding, self.padding, self.padding, self.padding])
        # We unfold into our kernel areas
        x = x.unfold(3, self.kernel_size[0], self.stride[0]) # Over W
        x = x.unfold(4, self.kernel_size[1], self.stride[1])  # Over H giving [N, samples, in_channels, H_fields, W_fields, H_kernel, W_kernel]
        # Then we multiply in our weights which are [samples, out_channels, in_samples, H_kernel, W_kernel]
        # This gives [N, samples, out_channels, H, W]
        x = torch.einsum('ijklmno,jpkno->ijplm',[x, self.weight])
        x = x + self.bias.unsqueeze(0).unsqueeze(3).unsqueeze(4)
        return x


class SVIMaxPool2D(nn.Module):
    """
    Expects
    :param x: [examples, samples, channels, H, W]
    :param kernel_size:  [H, W]
    :param stride: [H, W]
    :param padding: Not implemented
    :param dilation:
    :return:
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(SVIMaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        if stride == None:
            self.stride = kernel_size
        else:
            self.stride = stride
        if dilation != 1:
            raise NotImplementedError

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        x = x.unfold(3, self.kernel_size[0], self.stride[0])
        x = x.unfold(4, self.kernel_size[1], self.stride[1]) #  Now this is [examples, samples, channels, pooled_H, pooled_W, size_H, size_W)
        x = x.max(6)[0].max(5)[0]
        return x

    def extra_repr(self):
        s = ('pool_size={kernel_size}, stride={stride}, padding={padding}')
        return s.format(**self.__dict__)

class SVIGlobalMaxPool2D(nn.Module):
    """
        Expects
        :param x: [examples, samples, channels, H, W]
        :return: [examples, samples, channels]
        """

    def __init__(self):
        super(SVIGlobalMaxPool2D, self).__init__()


    def forward(self, x):
        x = x.max(4)[0].max(3)[0]
        return x


class SVIGlobalMeanPool2D(nn.Module):
    """
        Expects
        :param x: [examples, samples, channels, H, W]
        :return: [examples, samples, channels]
        """

    def __init__(self):
        super(SVIGlobalMeanPool2D, self).__init__()

    def forward(self, x):
        x = x.mean(4).mean(3)
        return x
