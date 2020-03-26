import os
import json
import argparse
import torch

# We import the following radial_layers so that we can initialize objects of the correct class using the config files
import data_loader.data_loaders as module_data_loaders
import radial_layers.loss as module_loss
import model.metric as module_metrics
import model.model as module_model

from trainer import Trainer  # object to manage training and validation loops
from utils.util import manage_seed  # helper functions

def main(config, resume):
    """
    Completes single optimization run.
    :param config: Dictionary from configuration file. See example in configs/
    :param resume: Path to checkpoint to resume from.
    :return: monitor_best, monitor last, monitor_best_se (the best metric measured, the final metric measured,
    the standard error of the best metric measured)
    """

    used_seed = manage_seed(config['seed'])  # You may want to log this with whatever tool you prefer

    # Setup data_loader instances
    config["data_loader"]["args"]["stage"] = "training"
    data_loader = getattr(module_data_loaders, config["data_loader"]["type"])(**config["data_loader"]["args"])
    config["data_loader"]["args"]["stage"] = "validation"
    valid_data_loader = getattr(module_data_loaders, config["data_loader"]["type"])(**config["data_loader"]["args"])
    # Build models
    model = getattr(module_model, config["arch"]["type"])(**config["arch"]["args"])
    model.summary()

    # Set the loss
    loss = getattr(module_loss, config["loss"]["type"])(**config["loss"]["args"])
    if hasattr(loss, "set_model"):
        # The ELBO loss needs to know the batch size to correctly balance factors
        loss.set_model(model, config)

    # Define the list of metric functions to use for training
    metrics = [getattr(module_metrics, met) for met in config['metrics']]

    # build optimizer.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(torch.optim, config["optimizer"]["type"])(trainable_params,
                                                                  **config["optimizer"]["args"])

    trainer = Trainer(model,
                      loss,
                      metrics,
                      optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader)


    monitor_best, monitor_last, monitor_best_se = trainer.train()

    return monitor_best, monitor_last, monitor_best_se

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file if one is provided
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
        # At the moment, preferred default behaviour is to fail here. Comment out assertion if you want
        # to use 'config.json' as a default.
        config = json.load(open('config.json'))
        path = os.path.join(config['trainer']['save_dir'], config['name'])

    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume)



