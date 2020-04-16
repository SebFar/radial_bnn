import torch
import torch.nn.functional as F


class Binary_cross_entropy():
    def compute_loss(self, y_input, y_target):
        return F.binary_cross_entropy_with_logits(y_input, y_target.type(torch.cuda.FloatTensor))

    def set_model(self, blank1, blank2):
        # Placeholder for compatibility with Elbo
        return


class Nll_Loss():

    def __init__(self):
        return

    def compute_loss(self, y_input, y_target):
        return 0, 0, F.nll_loss(y_input, y_target.squeeze())


class Mse_Loss():

    def __init__(self):
        return

    def compute_loss(self, y_input, y_target):
        return F.mse_loss(y_input, y_target)


class Elbo():

    def __init__(self,
                 binary,
                 regression):
        # In the case of binary classification we make different assumption about the input dimension
        self.binary = binary
        self.regression = regression
        self.writer = None
        return

    def set_model(self, model, config):
        self.model = model
        self.batch_size = config['data_loader']['args']['batch_size']

    def set_num_batches(self, num_batches):
        self.num_batches = num_batches

    def set_writer(self, writer):
        self.writer = writer

    def compute_loss(self, y_predicted, y_target):
        """
        Estimates the variational free energy loss function ELBO. Note that the KL divergence terms are computed
        within the variational mean-field layers, and we loop over them here.
        :param y_predicted: output of the forward pass
        :param y_target: target from training/test knowledge
        :return:
        """
        # The overall loss is
        # - ELBO = entropy_sum - cross_entropy_sum - negative_log_likelihood_sum
        # We calculate each separately. The first two depend on weights, the last on data.

        # This term accumulates the cross-entropy between the posterior and prior
        # L_\text{cross-entropy} in the paper
        # \int q(w) log[p(w)] dw
        # This is estimated using MC integration.
        cross_entropy_sum = 0

        # This term accumulates the entropy of the variational posterior
        # L_\text{entropy} in the paper
        # \int q(w) log[q(w)] dw
        # This is found analytically up to a constant and is shown in the paper to be
        # -\sum_i log[\sigma_i] + c
        # Where i sums over the weights. Which is up to a constant the same
        # as when w is distributed with a multivariate Gaussian
        entropy_sum = 0

        for module in self.model.modules():
            # Iterating over all radial_layers, including one representing the module as a whole.
            # So check if it supports the loss.
            if hasattr(module, "cross_entropy"):
                cross_entropy_sum += module.cross_entropy()
            if hasattr(module, "entropy"):
                entropy_sum += module.entropy()

        # Estimate the log likelihood of the data given the parameters
        # log(P(D|*w*))
        # Note that there is log softmax inside the model and that NLL loss performs elementwise mean by default
        # Instead, we want to mean over samples from the variational distirbution and sum over examples
        # y_input: Tensor predictions [examples, samples, classes]
        # target: Tensor of targets [examples]

        # First we add a samples dimension
        epoch_variational_samples = y_predicted.shape[1]
        y_target = y_target.unsqueeze(1)
        y_target = y_target.expand((-1, epoch_variational_samples))

        # In the case of regression we must estimate the target noise
        if self.regression:
            y_target = y_target.mean(dim=1)
            variance = y_target.var(dim=0)
            nll_tensor = (y_target - y_predicted.squeeze(dim=2).mean(dim=1)) ** 2 / (2 * variance)
        else:
            if self.binary:
                nll_tensor = F.binary_cross_entropy_with_logits(y_predicted.squeeze(dim=2), y_target.type(torch.cuda.FloatTensor), reduction='none')
            else:
                nll_tensor = F.nll_loss(y_predicted.permute(0, 2, 1), y_target, reduction="none")
        if len(nll_tensor.shape) > 1:
            # This should be taking the expectation over epsilon. We squeeze this in the binary case, so need to be
            # careful
            nll_tensor = torch.mean(nll_tensor, dim=1)
        nll_sum = torch.sum(nll_tensor)

        if self.writer is not None:
            self.writer.add_scalar('cross-entropy', cross_entropy_sum/self.num_batches)
            self.writer.add_scalar('entropy', entropy_sum/self.num_batches)
            self.writer.add_scalar('nll_loss', nll_sum)
        kl_divergence_estimated_over_batch = (cross_entropy_sum - entropy_sum) / self.num_batches

        return nll_sum / self.batch_size, kl_divergence_estimated_over_batch / self.batch_size
