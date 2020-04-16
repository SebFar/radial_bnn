# Radial Bayesian Neural Networks
Code to accompany the paper [Radial Bayesian Neural Networks: Beyond Discrete Support in Large-Scale Bayesian Deep Learning](https://arxiv.org/abs/1907.00865).

## Using the code
Implementations of pytorch layers supporting stochastic variational inference with Radial BNNs can be found in the folder `radial_layers/variational_bayes.py`.
These layers are compatible with the Elbo loss implemented in `radial_layers/loss.py`.
To drop this code into yours, you need only place the `radial_layers` folder into your pytorch code and instantiate layer objects imported from `variational_bayes.py`.

Layers inherit from the `SVI_Base` class, which instantiates stochastic variational inference weights and biases using the reparameterization trick.
At the moment, the available layers are `SVI_Linear`, `SVI_Conv2D`, `SVIMaxPool2D`, `SVIGlobalMaxPool2D`, and`SVIAverageMaxPool2D`.
Further contributions are very welcome.

### Requirements
This version of the code has been tested on pytorch version 1.4.0 with python 3.8.1 (though earlier iterations of the code worked in earlier versions of both). By default the code expects a cuda device, but can be modified to run on cpu.

## Example usage
This code-base provides two example usages.

`simple_example.py` provides a minimal example for training MNIST using a Radial BNN multi-layer perceptron.
It strips away advanced features and boilerplate.

The more advanced example, `train.py` loads configuration files from `configs` and builds trainer, model, loss, and optimizer objects based on the configuration.
For example, you can see `config_cnn_mnist.json` for an example that loads and trains a CNN on MNIST.
This includes features like allowing you to pre-train the means only using NLL loss (important for MFVI and not needed for Radial BNNs).

You can easily change the code to run standard mean-field variational inference with a multivariate Gaussian instead of Radial BNNs by changing the config file to set `arch["args"]["variational_distribution"] = "gaussian"` instead of `radial`.

For further details, issues are welcome, as are questions emailed to `sebastian.farquhar@cs.ox.ac.uk`.

## Citing this code
If you use this code, please cite us as:
```
@article{farquhar_radial_2020,
    author = {Sebastian Farquhar and Michael Osborne and Yarin Gal},
    title = {Radial Bayesian Neural Networks: Beyond Discrete Support in Large-Scale Bayesian Deep Learning},
    journal = {Proceedings of the 23rtd International Conference on Artificial Intelligence and Statistics},
    year = {2020}
}
```

## Acknowledgements
Much of the 'boilerplate' code of the example derives from the https://github.com/victoresque/pytorch-template 