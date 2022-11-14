import torch 


def MSE(tens):
    """Compute the mean of the squared (element-wise) tensor."""
    return torch.mean(tens**2)


def RMSE(tens):
    """Compute the root mean of the squared (element-wise) tensor."""
    return torch.sqrt(torch.mean(tens**2))


