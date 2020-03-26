import torch
import numpy as np

def accuracy(output, target):
    """
    Accuracy metric for deterministic model for classification.
    Args:
        output: Output Tensor with shape (examples, classes)
        target: Target Tensor with shape (examples)

    Returns: float accuracy

    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def binary_accuracy(output, target):
    """
        Accuracy metric for deterministic model for binary classification.
        Args:
            output: Output Tensor with shape (examples)
            target: Target Tensor with shape (examples)

        Returns: float accuracy

        """
    with torch.no_grad():
        pred = torch.gt(output, torch.zeros_like(output)).type(torch.cuda.LongTensor)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def variational_accuracy(y_predicted, y_target):
    """
    Finds the modal class over the number of samples and evaluates accuracy of that prediction
    Takes logits.

    :param y_predicted: Tensor [examples, samples from the variational posterior, class]
    :param y_target: Tensor [examples]
    :return: float accuracy
    """
    assert len(y_predicted) == len(y_target), "The output y should be the same length as the targets"
    # output of the model is a log_softmax (which is more efficient in the training loop), so we exponentiate
    y_predicted = torch.exp(y_predicted)
    class_prediction = torch.argmax(y_predicted, dim=2)
    modal_prediction, _ = torch.mode(class_prediction, dim=1)
    assert modal_prediction.shape[0] == y_predicted.shape[0], "arg maxes should not have changed the length"
    correct = torch.sum(modal_prediction == y_target)
    assert correct <= len(y_predicted), "Should not be more correct answers than examples."
    return (correct.type(torch.float32) / len(y_predicted)).cpu()


def binary_variational_accuracy(y_input, y_target):
    """
    As above, but for binary classification.
    :param y_input: Tensor [examples, samples from the variational posterior]
    :param y_target: Tensor [examples]
    :return:
    """
    assert len(y_input) == len(y_target), "The output y should be the same length as the targets"
    y_input = torch.sigmoid(y_input)
    class_prediction = y_input > 0.5
    modal_prediction, _ = torch.mode(class_prediction, dim=1)
    assert modal_prediction.shape[0] == y_input.shape[0], "arg maxes should not have changed the length"
    correct = torch.sum(modal_prediction.squeeze() == y_target.type(torch.cuda.ByteTensor))
    assert correct <= len(y_input), "Should not be more correct answers than examples."
    return (correct.type(torch.float32) / len(y_input)).cpu()


def _nll(y_input, y_target):
    """
        Finds the nll of the posterior distribution estimated by MC marginalization over epsilon
        :param y_input: Tensor [N, samples, class]
        :param y_target: Tensor [N]
        :return: the negative log likelihoods of predictions on each example
    """
    y_input = torch.exp(y_input)  # model output is log_softmax so we exponentiate
    y_posterior = torch.mean(y_input, dim=1)  # Average over all the samples to marginalize over epsilon
    # y_input is now [N, class]
    # We want the log-likelihood as a proper scoring rule
    likelihood = np.choose(y_target.cpu().numpy(),
                           y_posterior.cpu().numpy().T)  # Index the posterior by the true class [N]
    bump = 1e-25
    likelihood += bump  # We add a small constant to each term to avoid infinities
    nll = - np.log(likelihood)
    return nll

def mean_nll(y_input, y_target):
    return np.mean(_nll(y_input, y_target))


def _mi(y_input, y_target):
    """
    Estimates the mutual information between the inputs and the parameters of the model
    :param y_input: Expects the pre-exponential softmax output. Shape is [batch, samples, classes]
    :param y_target:
    :return:
    """
    # sum over the classes to get the Shannon entropy. Then we average over samples from the variational distribution
    # to estimate the empirical expected entropy
    expected_entropy = - torch.mean(torch.sum(y_input * torch.exp(y_input), dim=2), dim=1)
    # Now we average over samples from the variational distribution to get the expected probability for each class
    # for each example
    expected_p = torch.mean(torch.exp(y_input), dim=1)

    # We find the entropy of these expected probabilities by summing over the classes
    entropy_expected_p = - torch.sum(expected_p * torch.log(expected_p + 1e-25), dim=1)
    # Finally the MI is the difference between the two.
    mutual_information = entropy_expected_p - expected_entropy

    return mutual_information

def _binary_mi(y_input, y_target):
    """
    Estimates the mutual information between the inputs and the parameters of the model. Assumes output from the binary
    version of my model.
    :param y_input: Expects the pre-exponential softmax output. Shape is [batch, samples, 1]
    :param y_target:
    :return:
    """
    # sum over the classes to get the Shannon entropy. Then we average over samples from the variational distribution
    # to estimate the empirical expected entropy
    y_input = torch.sigmoid(y_input)
    expected_entropy = - torch.mean(y_input * torch.log(y_input + 1e-25) + (1 - y_input) * torch.log(1 - y_input + 1e-25), dim=1)
    expected_entropy = expected_entropy  # Remove the final dimension which is 1 anyhow.
    # Now we average over samples from the variational distribution to get the expected probability for each class
    # for each example
    expected_p = torch.mean(y_input, dim=1)

    # We find the entropy of these expected probabilities by summing over the classes
    entropy_expected_p = - ((expected_p * torch.log(expected_p + 1e-25)) + ((1-expected_p) * torch.log((1 - expected_p) + 1e-25)))
    # Finally the MI is the difference between the two.
    mutual_information = entropy_expected_p - expected_entropy

    return mutual_information

def mutual_information(y_input, y_target):
    return np.mean((_mi(y_input, y_target).cpu().numpy()))

def predictive_entropy(y_input, y_target):
    """
    Computes the entropy of predictions by the model
    :param y_input: Tensor [N, samples, class]
    :param y_target: Tensor [N] Not used here.
    :return: mean entropy over all examples
    """
    y_input = torch.exp(y_input)  # model output is log_softmax so we exponentiate
    y_posterior = torch.mean(y_input, dim=1)  # Average over all the samples to marginalize over epsilon
    # y_input is now [N, class]
    # We want the entropy of y_input
    epsilon = 1e-25
    y_posterior += epsilon  # We add a small constant to each term to avoid infinities
    entropy = - torch.mean(y_posterior * torch.log(y_posterior), dim=1)  # [N] entropy on each example
    return torch.mean(entropy).cpu().numpy()