import torch

from python_code.utils.config_singleton import Config

conf = Config()


def calculate_ber(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns the calculated ber of the prediction and the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    bits_acc = torch.mean(torch.eq(prediction, target).float()).item()
    return 1 - bits_acc

def calculate_error_rate(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns binary vector of correct (0) or wrong (1) prediction of the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    return 1 - torch.eq(prediction, target).float()