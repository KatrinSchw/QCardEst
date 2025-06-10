import math
import torch
from math import pi
import numpy as np
# Qiskit imports
import qiskit as qk
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator

# Qiskit Machine Learning imports
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector

from ML import layer
from ML import circuits
from ML.noiseModels import get_depolarising_noise_model


def classicalNN(settings={}, nInputs=4, nOutputs=15, norm=False):
    normLayer = layer.NormLayer()
    model = torch.nn.Sequential(
        torch.nn.Linear(nInputs, 64),
        torch.nn.Linear(64, nOutputs),
        torch.nn.Sigmoid(),
        #torch.nn.ReLU(),
        #normLayer
    )
    return model


def vqc(settings: dict, nInputs: int, nOutputs: int, norm=True) -> torch.nn.Module:
    """
    Create a VQC as a pytorch model
    """
    # convert single encoding gate into a list
    if type(settings["encoding"]) is not list:
        settings["encoding"] = [settings["encoding"]]

    # Calculate number of qubits
    nQubitsOut = math.ceil(math.log2(nOutputs))
    nQubits = max(nInputs, nQubitsOut)

    # Generate the Parametrized Quantum Circuit
    qc = circuits.parametrizedCircuit(nQubits, settings)

    # Fetch the parameters from the circuit and divide them in Inputs (X) and Trainable Parameters (params)
    divider = nQubits * len(settings["encoding"])
    X = list(qc.parameters)[:divider]
    params = list(qc.parameters)[divider:]

    # Select a quantum backend to run the simulation of the quantum circuit
    if settings["noise"] > 0:
        depol_error_prob = settings["noise"]
        noise_model = get_depolarising_noise_model(depol_error_prob)
        backend = AerSimulator(noise_model=noise_model)
    else:
        backend = qk.Aer.get_backend('aer_simulator_statevector')
    qi = QuantumInstance(backend, shots=10000)
    if 'GPU' in backend.available_devices():
        backend.set_options(device='GPU')

    # Create a Quantum Neural Network object
    qnn = CircuitQNN(qc, input_params=X, weight_params=params, quantum_instance=qi)

    # Connect to PyTorch
    initialWeights = (2 * pi * np.random.rand(qnn.num_weights) - pi)
    quantumNN = TorchConnector(qnn, initialWeights)

    # build model
    model = torch.nn.Sequential()

    # pad input with zeros to number of qubits
    if (nQubitsOut > nInputs):
        paddingLayer = torch.nn.ConstantPad1d((0, nQubitsOut - nInputs), 2.5 * math.pi)
        model.append(paddingLayer)

    # add quantum layer
    model.append(quantumNN)

    # reduce number of states to number of outputs
    model.append(layer.ReshapeSumLayer(nOutputs))

    # add norm layer. Assures one value is 1.0
    if norm:
        model.append(layer.NormLayer())

    return model
