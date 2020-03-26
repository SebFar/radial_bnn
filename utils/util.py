import os
import torch
import numpy as np

def manage_seed(seed):
    """
    If we want a random seed, randomizes and notes. If you want a fixed seed, uses that.
    :param seed: "random" for a random seed or an int.
    :return: The seed used.
    """
    if seed == "random":
        # In this case we want the seed to be different each time for the same config.
        # But we want to log which seed it was
        np.random.seed()
        seed_to_use = np.random.randint(10000)
    else:
        assert isinstance(seed, int), "Unless seed is the string 'random' it must be an integer which will be used as a seed."
        seed_to_use = seed
    np.random.seed(seed_to_use)
    torch.manual_seed(seed_to_use)
    torch.cuda.manual_seed(seed_to_use)
    return seed_to_use

def ensure_dir(path):
    """
    Makes sure that a directory exists. If it doesn't, it makes it.
    :param path:
    :return: True if it made the directory, False if it did not
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            return True
    except IOError:
        print("Error: cannot make directory at %r" % path)
    else:
        return False