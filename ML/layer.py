import torch


class ReshapeSumLayer(torch.nn.Module):
    """
    Reduce to 'target' number of values by suming
    """

    def __init__(self, target: int):
        super().__init__()
        self.target = target

    def forward(self, x):
        size = x.size()
        size = size[:-1] + (self.target, -1)
        result = torch.reshape(x, size)
        return result.sum(-1)


class NormLayer(torch.nn.Module):
    """
    Scale the value vectors thus that the largest value ist equal to 1
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        result = x / x.max()
        return result


class CutOutputLayer(torch.nn.Module):
    """
    Reduce to 'target' number of values by cutting of additional values
    """

    def __init__(self, target):
        super().__init__()
        self.nOut = target

    def forward(self, x):
        return torch.narrow(x, -1, 0, self.nOut)
