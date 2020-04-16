import torch
from torch.nn import Module
from torch.nn import functional as F

from radial_layers.variational_bayes import SVI_Linear
from radial_layers.loss import Elbo
from data_loader.data_loaders import MNISTDataLoader

# Build the model, using pre-made SVI layers


class SVI_MNIST_MLP(Module):
    """
    Very basic stochastic variational inference MLP for classification.

    You can create this from JSON args in the full train.py pipeline.
    """

    def __init__(self):
        super(SVI_MNIST_MLP, self).__init__()
        initial_rho = -4  # This is a reasonable value, but not very sensitive.
        initial_mu_std = (
            "he"  # Uses Kaiming He init. Or pass float for a Gaussian variance init.
        )
        variational_distribution = "radial"  # You can use 'gaussian' to do normal MFVI.
        prior = {
            "name": "gaussian_prior",
            "sigma": 1.0,
            "mu": 0,
        }  # Just a unit Gaussian prior.
        self.first_layer = SVI_Linear(
            in_features=784,
            out_features=200,
            initial_rho=initial_rho,
            initial_mu=initial_mu_std,
            variational_distribution=variational_distribution,
            prior=prior,
        )
        self.hidden_layer = SVI_Linear(
            in_features=200,
            out_features=200,
            initial_rho=initial_rho,
            initial_mu=initial_mu_std,
            variational_distribution=variational_distribution,
            prior=prior,
        )
        self.last_layer = SVI_Linear(
            in_features=200,
            out_features=10,
            initial_rho=initial_rho,
            initial_mu=initial_mu_std,
            variational_distribution=variational_distribution,
            prior=prior,
        )

    def forward(self, x):
        # Input has shape [examples, samples, height, width]
        x = x.view(x.size()[0], x.size()[1], -1)
        x = F.relu(self.first_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = F.log_softmax(self.last_layer(x), dim=-1)
        return x


# Now call a standard training loop as you normally would!

train_loader = MNISTDataLoader("data", stage="training")
model = SVI_MNIST_MLP()

# We need to set a couple extra values to make sure that the ELBO is calculated right.
loss = Elbo(binary=False, regression=False)
loss.set_model(model, {"data_loader": {"args": {"batch_size": 16}}})
loss.set_num_batches(len(train_loader))

optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
device = "cuda"
log_interval = 100
variational_samples = 8  # Sets number of samples from model to use per forward pass

# This implementation ignores options like pretraining means or advanced configs
# that are demonstrated in the main train.py file
model = model.to(device)
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # We tile the data to efficiently parallize forward passes through multiple
        # samples from the variational posterior
        data = data.expand((-1, variational_samples, -1, -1))

        optimizer.zero_grad()
        output = model(data)  # Output has shape [examples, samples, classes]
        nll_loss, kl_term = loss.compute_loss(output, target)
        batch_loss = nll_loss + kl_term
        batch_loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    batch_loss.item(),
                )
            )
