import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from radial_layers.variational_bayes import SVI_Linear, SVIConv2D, SVIMaxPool2D, SVIGlobalMaxPool2D, SVIGlobalMeanPool2D


class SVI_Regression_MLP(BaseModel):
    """
    Basic stochastic variational inference MLP for regression.

    Config should be structured:
    "arch": {
        "type": "SVI_Regression_MLP",
        "args": {
            "in_features": 784,  # input dimensionality
            "hidden_features": [40, 40],  # list of unit width for each hidden layer
            "initial_rho": -6,  # initial \sigma parameter will be softplus(\rho)
            "initial_mu": "he",  # or float to manually set Gaussian initialization standard deviation
            "variational_distribution": "radial"  # or "gaussian",
            "prior": {
                "name": "gaussian",
                "sigma": 1,
                "mu": 0
            }
        }
    }
    """
    def __init__(self,
                 in_features,
                 hidden_features,
                 initial_rho,
                 initial_mu_std,
                 variational_distribution,
                 prior):
        super(SVI_Regression_MLP, self).__init__()

        self.first_layer = SVI_Linear(in_features,
                                      hidden_features[0],
                                      initial_rho,
                                      initial_mu_std,
                                      variational_distribution,
                                      prior)

        self.hidden_layers = nn.ModuleList()
        for idx, hidden_out in enumerate(hidden_features[1:]):
            self.hidden_layers.append(SVI_Linear(hidden_features[idx],
                                                 hidden_out,
                                                 initial_rho,
                                                 initial_mu_std,
                                                 variational_distribution,
                                                 prior))
        self.last_layer = SVI_Linear(hidden_features[-1],
                                     1,
                                     initial_rho,
                                     initial_mu_std,
                                     variational_distribution,
                                     prior)

    def forward(self, x):
        # Input expects an image with shape [examples, samples, height, width]
        x = x.view(x.size()[0], x.size()[1], -1)
        x = F.relu(self.first_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.last_layer(x)
        return x


class SVI_Classification_MLP(BaseModel):
    """
    Basic stochastic variational inference MLP for classification.

    Config should be structured:
    "arch": {
        "type": "SVI_Regression_MLP",
        "args": {
            "in_features": 784,  # input dimensionality
            "hidden_features": [40, 40],  # list of unit width for each hidden layer
            "out_features": 10,  # Number of output classes
            "initial_rho": -6,  # initial \sigma parameter will be softplus(\rho)
            "initial_mu": "he",  # or float to manually set Gaussian initialization standard deviation
            "variational_distribution": "radial"  # or "gaussian",
            "prior": {
                "name": "gaussian",
                "sigma": 1,
                "mu": 0
            }
        }
    }
    """
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 initial_rho,
                 initial_mu_std,
                 variational_distribution,
                 prior):
        super(SVI_Classification_MLP, self).__init__()

        self.first_layer = SVI_Linear(in_features,
                                      hidden_features[0],
                                      initial_rho,
                                      initial_mu_std,
                                      variational_distribution,
                                      prior)
        self.hidden_layers = nn.ModuleList()
        for idx, hidden_out in enumerate(hidden_features[1:]):
            self.hidden_layers.append(SVI_Linear(hidden_features[idx],
                                                 hidden_out,
                                                 initial_rho,
                                                 initial_mu_std,
                                                 variational_distribution,
                                                 prior))
        self.last_layer = SVI_Linear(hidden_features[-1],
                                     out_features,
                                     initial_rho,
                                     initial_mu_std,
                                     variational_distribution,
                                     prior)

    def forward(self, x):
        # Input has shape [examples, samples, height, width]
        x = x.view(x.size()[0], x.size()[1], -1)
        x = F.relu(self.first_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = F.log_softmax(self.last_layer(x), dim=-1)
        return x


class SVI_Conv_Classifier(BaseModel):
    """
    Simple example stochastic variational inference CNN for classification.

    Config should be structured:
    "arch": {
        "type": "SVI_Conv_Classifier",
        "args": {
            "in_channels": 1,  # input channels
            "out_features": 10,  # output classes
            "initial_rho": -6,  # initial \sigma parameter will be softplus(\rho)
            "initial_mu": "he",  # or float to manually set Gaussian initialization standard deviation
            "variational_distribution": "radial"  # or "gaussian",
            "prior": {
                "name": "gaussian",
                "sigma": 1,
                "mu": 0
            }
        }
    }
    """
    def __init__(self,
                 in_channels,
                 out_features,
                 initial_rho,
                 initial_mu_std,
                 variational_distribution,
                 prior,):
        super(SVI_Conv_Classifier, self).__init__()
        self.first_conv = SVIConv2D(in_channels,
                                    32,
                                    [3, 3],
                                    variational_distribution,
                                    prior,
                                    initial_rho,
                                    initial_mu_std,
                                    padding=2)
        self.max_pool = SVIMaxPool2D((2, 2))
        self.second_conv = SVIConv2D(32,
                                     64,
                                     [3, 3],
                                     variational_distribution,
                                     prior,
                                     initial_rho,
                                     initial_mu_std,
                                     padding=1)
        self.third_conv = SVIConv2D(64,
                                    128,
                                    [3, 3],
                                    variational_distribution,
                                    prior,
                                    initial_rho,
                                    initial_mu_std,
                                    padding=1)

        # self.global_mean = SVIGlobalMeanPool2D()
        # self.global_max = SVIGlobalMaxPool2D()

        self.fc1 = SVI_Linear(128 * 3 * 3,
                              64,
                              initial_rho,
                              initial_mu_std,
                              variational_distribution,
                              prior)
        self.last_layer = SVI_Linear(64,
                                     out_features,
                                     initial_rho,
                                     initial_mu_std,
                                     variational_distribution,
                                     prior)

    def forward(self, x):
        # Input has shape [examples, samples, channels, height, width]
        x = self.first_conv(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = F.relu(self.second_conv(x))
        x = self.max_pool(x)
        x = F.relu(self.third_conv(x))
        x = self.max_pool(x)
        s = x.shape
        x = torch.reshape(x, (s[0], s[1], -1))
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.last_layer(x), dim=-1)
        return x

class SVI_VGG16_Retinopathy_Conv_Model(BaseModel):

    def __init__(self, in_channels, conv_channel_base, out_features, initial_rho, initial_mu_std, variational_distribution, prior):
        super(SVI_VGG16_Retinopathy_Conv_Model, self).__init__()
        self.activation = torch.nn.LeakyReLU(0.2)
        self.block_1_conv_1 = SVIConv2D(in_channels, conv_channel_base, [3, 3], variational_distribution, prior, initial_rho, initial_mu_std,
                                        padding=1, stride=(2,2))
        self.block_1_conv_2 = SVIConv2D(conv_channel_base, conv_channel_base, [3, 3], variational_distribution, prior,
                                        initial_rho, initial_mu_std,
                                        padding=1, stride=(1, 1))
        self.max_pool = SVIMaxPool2D((2, 2), stride=(2, 2), padding=0)
        self.block_2_conv_1 = SVIConv2D(conv_channel_base, conv_channel_base * 2, [3, 3], variational_distribution, prior, initial_rho,
                                        initial_mu_std,
                                        padding=1, stride=(1, 1))
        self.block_2_conv_2 = SVIConv2D(conv_channel_base * 2, conv_channel_base * 2, [3, 3], variational_distribution, prior,
                                        initial_rho,
                                        initial_mu_std,
                                        padding=1, stride=(1, 1))

        self.block_3_conv_1 = SVIConv2D(conv_channel_base * 2, conv_channel_base * 4, [3, 3], variational_distribution, prior,
                                        initial_rho,
                                        initial_mu_std,
                                        padding=1, stride=(1, 1))
        self.block_3_conv_2 = SVIConv2D(conv_channel_base * 4, conv_channel_base * 4, [3, 3], variational_distribution, prior,
                                        initial_rho,
                                        initial_mu_std,
                                        padding=1, stride=(1, 1))
        self.block_3_conv_3 = SVIConv2D(conv_channel_base * 4, conv_channel_base * 4, [3, 3], variational_distribution, prior,
                                        initial_rho,
                                        initial_mu_std,
                                        padding=1, stride=(1, 1))

        self.block_4_conv_1 = SVIConv2D(conv_channel_base * 4, conv_channel_base * 8, [3, 3], variational_distribution, prior,
                                        initial_rho,
                                        initial_mu_std,
                                        padding=1, stride=(1, 1))

        self.block_4_conv_2 = SVIConv2D(conv_channel_base * 8, conv_channel_base * 8, [3, 3], variational_distribution,
                                        prior,
                                        initial_rho,
                                        initial_mu_std,
                                        padding=1, stride=(1, 1))

        self.block_4_conv_3 = SVIConv2D(conv_channel_base * 8, conv_channel_base * 8, [3, 3], variational_distribution,
                                        prior,
                                        initial_rho,
                                        initial_mu_std,
                                        padding=1, stride=(1, 1))

        self.block_5_conv_1 = SVIConv2D(conv_channel_base * 8, conv_channel_base * 8, [3, 3], variational_distribution, prior,
                                        initial_rho,
                                        initial_mu_std,
                                        padding=1, stride=(1, 1))
        self.block_5_conv_2 = SVIConv2D(conv_channel_base * 8, conv_channel_base * 8, [3, 3], variational_distribution, prior,
                                        initial_rho,
                                        initial_mu_std,
                                        padding=1, stride=(1, 1))
        self.block_5_conv_3 = SVIConv2D(conv_channel_base * 8, conv_channel_base * 8, [3, 3], variational_distribution,
                                        prior,
                                        initial_rho,
                                        initial_mu_std,
                                        padding=1, stride=(1, 1))

        self.global_mean = SVIGlobalMeanPool2D()
        self.global_max = SVIGlobalMaxPool2D()
        self.out_features = out_features
        self.last_layer = SVI_Linear(conv_channel_base * 16,
                                     out_features,
                                     initial_rho,
                                     initial_mu_std,
                                     variational_distribution,
                                     prior)

    def forward(self, x):
        # Input has shape [examples, samples, channels, height, width]

        x = self.block_1_conv_1(x)
        x = self.activation(x)
        x = self.block_1_conv_2(x)
        x = self.activation(x)

        x = self.max_pool(x)

        x = self.block_2_conv_1(x)
        x = self.activation(x)
        x = self.block_2_conv_2(x)
        x = self.activation(x)

        x = self.max_pool(x)

        x = self.block_3_conv_1(x)
        x = self.activation(x)
        x = self.block_3_conv_2(x)
        x = self.activation(x)
        x = self.block_3_conv_3(x)
        x = self.activation(x)

        x = self.max_pool(x)

        x = self.block_4_conv_1(x)
        x = self.activation(x)
        x = self.block_4_conv_2(x)
        x = self.activation(x)
        x = self.block_4_conv_3(x)
        x = self.activation(x)

        x = self.max_pool(x)

        x = self.block_5_conv_1(x)
        x = self.activation(x)
        x = self.block_5_conv_2(x)
        x = self.activation(x)
        x = self.block_5_conv_3(x)
        x = self.activation(x)

        x_1 = self.global_mean(x)
        x_2 = self.global_max(x)

        x = torch.cat((x_1, x_2), dim=2)
        x = self.last_layer(x)

        if  self.out_features > 1:
            x = F.log_softmax(x, dim=-1)
        elif self.out_features == 1:
            x = x # We do binary_cross_entropy_with_logits and predictions must take sigmoid before using
        return x