import torch

from abc import ABC, abstractmethod

from ML.layer import CutOutputLayer

"""
Here interpretations of the probability output of a VQC for Regression are defined

Some interpretation do post-processing after the model
Most add a classical layer to the model
"""

# defines the interpretation of a ML model
class Interpretation(ABC):

    nInputs = 1
    "Number of required inputs"

    model: torch.nn.Module
    "Model the interpretation should be applied to"

    def __init__(self, model, env) -> None:
        self.model = model

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)

    def setModel(self, model):
        self.model = model

    def predict(self, prediction: torch.Tensor):
        """
        Allows a prediction other than the output of the model
        """
        return prediction

    def loss(self, prediction: torch.Tensor, compare):
        """
        Calculate a loss used gradient based optimization
        """
        return (self.predict(prediction) - compare)**2


class LinearScale(Interpretation):
    """
    Use a single probability and a classical scalar
    """

    nInputs = 2

    def __init__(self, model, env) -> None:
        self.cutLayer = CutOutputLayer(1)
        self.scaleLayer = torch.nn.Linear(1, 1, bias=False)
        self.scaleLayer.weight.data.fill_(env.maxExpected())
        self.model = torch.nn.Sequential(model, self.cutLayer, self.scaleLayer)

    def setModel(self, model):
        self.model = torch.nn.Sequential(model, self.cutLayer, self.scaleLayer)


class RationalLayer(torch.nn.Module):

    episolon = 0.0001
    "Prevent a division by zero"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.t()
        return (x[0] + self.episolon) / (x[1] + self.episolon)


class Rational(Interpretation):
    """
    Use the ratio between two probabilities
    """
    nInputs = 2

    def __init__(self, model, env) -> None:
        self.ratioLayer = RationalLayer()
        self.model = torch.nn.Sequential(model, self.ratioLayer)

    def setModel(self, model):
        self.model = torch.nn.Sequential(model, self.ratioLayer)


class RationalLogLayer(torch.nn.Module):

    episolon = 0.0001
    "Prevent a division by zero"
  
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log((x[0] + self.episolon) / (x[1] + self.episolon))


class RationalLog(Interpretation):
    """
    Use the log of the ration between two probabilities
    """
    nInputs = 2

    def __init__(self, model, env) -> None:
        self.ratioLayer = RationalLogLayer()
        self.model = torch.nn.Sequential(model, self.ratioLayer)

    def setModel(self, model):
        self.model = torch.nn.Sequential(model, self.ratioLayer)


class PlaceValue(Interpretation):

    def predict(self, prediction: torch.Tensor):
        return super().predict(prediction)


class NoChange(Interpretation):
    """
    Always returns 0.0
    """

    def predict(self, prediction: torch.Tensor):
        return torch.tensor(0.0, requires_grad=True)


class SecondValueThreshold(Interpretation):
    """
    For each qubit affecting the output a second qubit is used to control if the output should be affected
    Two such controlled qubits are combined by subtraction
    """
    nInputs = 4

    def __init__(self, model, env) -> None:
        newModel = self.model = torch.nn.Sequential(model, ThresholdControlledScalarLayer())
        super().__init__(newModel, env)

    def setModel(self, model):
        self.model = torch.nn.Sequential(model, ThresholdControlledScalarLayer())


class ThresholdControlledScalarLayer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # create scalar parameter
        self.scalar = torch.nn.Parameter(torch.tensor(1.0))

    def calculate(self, x):
        posChange = 1 + torch.relu(x[0] - 0.25) * x[1] * self.scalar**2
        negChange = 1 + torch.relu(x[2] - 0.25) * x[3] * self.scalar**2
        result = posChange - negChange
        return result

    def forward(self, x):
        # check if data is batch or single entry
        if (len(x.size()) == 1):
            result = self.calculate(x)
        else:
            result = torch.vmap(self.calculate)(x)
        return result


class SecondValueRatioThreshold(Interpretation):
    """
    For each qubit affecting the output a second qubit is used to control if the output should be affected
    Two such controlled qubits are combined by division
    """
    nInputs = 4

    def __init__(self, model, env) -> None:
        newModel = self.model = torch.nn.Sequential(model, ThresholdControlledScalarRatioLayer())
        super().__init__(newModel, env)

    def setModel(self, model):
        self.model = torch.nn.Sequential(model, ThresholdControlledScalarRatioLayer())


class ThresholdControlledScalarRatioLayer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(1.0))

    def calculate(self, x):
        posChange = 1 + torch.relu(x[0] - 0.25) * x[1] * self.scalar**2
        negChange = 1 + torch.relu(x[2] - 0.25) * x[3] * self.scalar**2
        result = posChange / negChange
        return result

    def forward(self, x):
        # check if data is batch or single entry
        if (len(x.size()) == 1):
            result = self.calculate(x)
        else:
            result = torch.vmap(self.calculate)(x)
        return result


class PlaceValueLayer(torch.nn.Module):

    start = 0

    def __init__(self, length=4, negativ=False, scalar=1):
        super().__init__()
        self.negativ = negativ
        self.length = length
        if negativ:
            self.numExp = int(length / 2)
        else:
            self.numExp = length
        if scalar == None:
            self.scalar = torch.nn.Parameter(torch.rand(()))
        else:
            self.scalar = torch.nn.Parameter(torch.tensor(scalar))
        assert self.scalar > 0

    def forward(self, x):
        exponent = torch.linspace(self.start, self.numExp - 1 - self.start, self.numExp)
        base = torch.full((1, self.numExp), 1 + self.scalar.item())
        factors = torch.pow(base, exponent)
        if self.negativ:
            factors = torch.cat((factors, -factors), dim=1)
        return torch.sum(x * factors, 1)


class PlaceValueSystem(Interpretation):


    def __init__(self, model, env, length=4, negativ=False, scalar=1.0) -> None:
        newModel = self.model = torch.nn.Sequential(model, PlaceValueLayer(length, negativ, scalar))
        super().__init__(newModel, env)
        self.nInputs = length
        self.scalar = scalar
        self.negativ = negativ

    def setModel(self, model):
        self.model = torch.nn.Sequential(model, PlaceValueLayer(self.nInputs, self.negativ, self.scalar))


def fromString(s, model, env) -> Interpretation:
    tmp = s.split(",")
    name = tmp[0]
    if len(tmp) > 1:
        arg = tmp[1]
    else:
        arg = None
    if name in ["linear"]:
        return LinearScale(model, env)
    if name in ["rational"]:
        return Rational(model, env)
    if name in ["rationalLog", "logRational"]:
        return RationalLog(model, env)
    if name in ["constant"]:
        return NoChange(model, env)
    if name in ["threshold"]:
        return SecondValueThreshold(model, env)
    if name in ["thresholdRatio"]:
        return SecondValueRatioThreshold(model, env)
    if name in ["PlaceValue", "PlaceValuePos", "placeValue", "placeValuePos"]:
        if arg == None:
            return PlaceValueSystem(model, env, negativ=False)
        else:
            return PlaceValueSystem(model, env, length=int(arg), negativ=False)
    if name in ["PlaceValueNeg", "placeValueNeg", "PlaceValueNegativ", "placeValueNegativ"]:
        if arg == None:
            return PlaceValueSystem(model, env, negativ=True)
        else:
            return PlaceValueSystem(model, env, length=int(arg), negativ=True)

    return LinearScale(model, env)
