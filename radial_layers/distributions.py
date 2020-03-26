import torch

# Priors

def gaussian_prior(name,
                   log2pi,
                   mu,
                   sigma,
                   device):
    """
    Args:
        *args: {"mu": , "sigma":, "log2pi"}

    Returns: log_gaussian_pdf that takes a weight of arbitrary shape

    """
    if mu == 0 and sigma == 1:
        # We handle this case slightly differently as it is common and can be made more efficient
        def log_gaussian_pdf(x):
            x = x.view(x.shape[0], -1)
            return - log2pi * x.shape[1] / 2 - torch.sum(x**2) / 2.
        return log_gaussian_pdf
    else:
        mu_tensor = torch.tensor(mu, requires_grad=False, dtype=torch.float32, device=device)
        sigma_tensor = torch.tensor(sigma, requires_grad=False, dtype=torch.float32, device=device)
        two_sigma_squared = 2 * (sigma_tensor ** 2)
        log_sigma = torch.log(sigma_tensor)

        def log_gaussian_pdf(x):
            x = x.view(x.shape[0], -1)
            log_pd = - log2pi * x.shape[1] / 2
            log_pd = log_pd - torch.sum((x - mu_tensor) ** 2, dim=1) / two_sigma_squared
            log_pd = log_pd - log_sigma * x.shape[1] / 2
            return log_pd

        return log_gaussian_pdf

# Sampling noise distributions

def radial(size):
    """
    Creates a distribution that is unit Gaussian along r and uniform over \theta.

    :param size: The size of the weight distribution to be generated.
                    Zeroth dimension is variational samples.
                    1+ dimensions are the weight for each sample from the variational distribution.
                    The same weight is applied to each example in a batch.
    :return: noise distribution
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # First we find a random direction (\epsilon_{\text{MFVI}} in equation (3) on page 4)
    epsilon_mfvi = torch.randn(size, device=device)

    # Then we pick a distance (r in equation (3) on page 4)
    distance = torch.randn((size[0]), device=device)

    # Then we normalize each variational sample independently
    if len(size) == 2:
        normalizing_factor = torch.norm(epsilon_mfvi.view(size[0], -1), p=2, dim=1).unsqueeze(1)
        distance = distance.unsqueeze(1)
    elif len(size) == 3:
        normalizing_factor = torch.norm(epsilon_mfvi.view(size[0], -1), p=2, dim=1).unsqueeze(1).unsqueeze(1)
        distance = distance.unsqueeze(1).unsqueeze(1)
    elif len(size) == 5:
        # Here we have a CNN with dimensions (var samples, out_channels, in_channels, kernel, kernel)
        normalizing_factor = torch.norm(epsilon_mfvi.view(size[0], -1), p=2, dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(
            1).unsqueeze(1)
        distance = distance.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    else:
        raise ValueError("Number of dimensions for epsilon not expected. Are you sure you wanted size {}".format(size))

    direction = epsilon_mfvi / normalizing_factor
    epsilon_radial = direction * distance
    return epsilon_radial


def gaussian(size):
    """
    Returns a tensor of random epsilon using the default gaussian unit distribution
    :param size: shape of tensor to return (tuple)
    :return: FloatTensor of Size
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    epsilon_mfvi = torch.randn(size,
                    device=device)
    return epsilon_mfvi