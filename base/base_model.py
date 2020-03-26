import logging
import torch
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def track_params(self, param_name):
        """

        :param param_name: should be either 'weight_rhos' or 'weight_mus'
        :return:
        """
        params = []
        for module in self.modules():
            if param_name == 'weight_sigmas':
                call_name = 'weight_rhos'
            else:
                call_name = param_name
            if hasattr(module, call_name):
                params_tensor = getattr(module, call_name)
                if call_name == "weight_rhos":
                    params_tensor = module._rho_to_sigma(params_tensor)
                [params_median, params_mean, params_max, params_min] = [torch.median(params_tensor),
                                                                    torch.mean(params_tensor),
                                                                    torch.max(params_tensor),
                                                                    torch.min(params_tensor)]
                params.append([params_median, params_mean, params_max, params_min])
        return params

    def pretraining(self, new_pretraining):
        layer_counter = 0
        for layer in self.modules():
            if hasattr(layer, 'is_pretraining'):
                layer_counter += layer.is_pretraining(new_pretraining)
        return layer_counter
