import torch 



def logit(x):
    """logit function. If x is too close from 1, we set the result to 0.
    performs logit element wise."""
    return torch.nan_to_num(torch.log(x / (1 - x)), nan=0, neginf=0, posinf=0)



def log_stirling(n):
    """Compute log(n!) even for n large. We use the Stirling formula to avoid
    numerical infinite values of n!.
    Args:
         n: torch.tensor of any size.
    Returns:
        An approximation of log(n_!) element-wise.
    """
    n_ = n + (n == 0)  # Replace the 0 with 1. It doesn't change anything since 0! = 1!
    return torch.log(torch.sqrt(2 * np.pi * n_)) + n_ * \
        torch.log(n_ / math.exp(1))  # Stirling formula

