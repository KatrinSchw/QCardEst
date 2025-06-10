import math

# Qiskit imports
import qiskit as qk
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import TwoLocal

from ML.circuitsSukinSim_et_al import fromNameSukim


class VQCsettings:
    """
    Class containing the settings for creating a VQC
    """

    reuploading = False
    reps = 2
    insert_barriers = True
    meas = False
    calc: str = "xyz"
    entangleGate = "cz"
    entangleType = "circular"

    def __init__(self, settings={}) -> None:
        if "reuploading" in settings:
            self.reuploading = settings["reuploading"]
        if "reps" in settings:
            self.reps = settings["reps"]
        if "calc" in settings:
            self.calc = settings["calc"]
            self.calcGates = stringToCircuitList(settings["calc"])
        if "entangle" in settings:
            self.entangleGate = settings["entangle"]
        if "entangleType" in settings:
            self.entangleType = settings["entangleType"]
        self.encodingGates = settings["encoding"]


def encodingCircuit(inputs: ParameterVector, nQubits: int, gates=["rx"]) -> qk.QuantumCircuit:
    """
    Create encoding layer of a VQC
    """
    if len(inputs) != nQubits * len(gates):
        raise ValueError("Expected " + str(nQubits * len(gates)) + " inputs, but got " + str(len(inputs)))

    qc = qk.QuantumCircuit(nQubits)

    # Encode data with a rotation depending on parameter gate
    for i in range(nQubits):
        for gi, gate in enumerate(gates):
            index = i * len(gates) + gi
            if gate == "rx" or gate == "x":
                qc.rx(inputs[index], i)
            if gate == "ry" or gate == "y":
                qc.ry(inputs[index], i)
            if gate == "rz" or gate == "z":
                qc.rz(inputs[index], i)
    return qc


def stringToCircuitList(string: str) -> list[str]:
    """
    Converts e.g. "xz" into ["rx","rz"]
    """
    result = []
    for c in string:
        result.append("r" + c)
    return result


# create a paramterized circuit (VQC)
def parametrizedCircuit(nQubits: int, settings: dict):

    vqcConf = VQCsettings(settings)
    qc = qk.QuantumCircuit(nQubits)

    # Define a vector containg Inputs as parameters (*not* to be optimized)
    inputs = ParameterVector('x', nQubits * len(vqcConf.encodingGates))

    if not vqcConf.reuploading:
        # Encode classical input data
        qc.compose(encodingCircuit(inputs, nQubits=nQubits, gates=vqcConf.encodingGates), inplace=True)
        qc.barrier()

        # Variational circuit
        processing = variationalLayers(nQubits, vqcConf, maxLayers=vqcConf.reps)
        qc.compose(processing, inplace=True)

    else:  #Reuploading
        # TODO: Implement for general circuit
        # Define a vector containng variational parameters
        theta = ParameterVector('theta', len(vqcConf.calcGates) * nQubits * vqcConf.reps)

        # Iterate for a number of repetitions
        for rep in range(vqcConf.reps):

            # insert a reuploading layer every 4 layers
            if rep % 4 == 0:
                qc.compose(encodingCircuit(inputs, nQubits=nQubits, gates=vqcConf.encodingGates), inplace=True)

            # Variational circuit (does the same as TwoLocal from Qiskit)
            for qubit in range(nQubits):
                i = 0
                for gate in vqcConf.calcGates:
                    layerOffset = len(vqcConf.calcGates) * nQubits * rep
                    index = qubit + layerOffset + i * nQubits
                    if gate == "rx":
                        qc.rx(theta[index], qubit)
                    if gate == "ry":
                        qc.ry(theta[index], qubit)
                    if gate == "rz":
                        qc.rz(theta[index], qubit)
                    i += 1

            # Add entanglers (this code is for a circular entangler)
            qc.cx(nQubits - 1, 0)
            for qubit in range(nQubits - 1):
                qc.cx(qubit, qubit + 1)

    #circuit_drawer(qc, output='mpl', filename="circuit.png")
    return qc


def variationalLayers(nQubits, settings: VQCsettings, maxDepth=math.inf, maxParam=math.inf, maxLayers=math.inf) -> qk.QuantumCircuit:
    """
        Creates a VQC by adding as many layers as the three limiting factors allow
        - maxDepth : the maximal depth the circuit can have
        - maxParam : the maximal number of parameters the circuit can use
        - maxLayers : the maximal number of layers
        At least one of these three limiting factors has to be set
    """
    if math.isinf(maxDepth) and math.isinf(maxParam) and math.isinf(maxLayers):
        raise ValueError("At least one value for maxDepth, maxParam or maxLayers has to be passed")
    nLayers = 0
    processing = qk.QuantumCircuit()

    # extend circuit until one of the limits is reached
    while (True):
        # create a new circuit
        if settings.calc.startswith("ansatzSukim"):
            if nQubits != 4:
                raise NotImplementedError("Sukim circuit are currently only implemented for 4 qubits")
            newProcessing, params = fromNameSukim(settings.calc, nLayers)
        else:
            newProcessing = TwoLocal(nQubits, settings.calcGates, settings.entangleGate, settings.entangleType, reps=nLayers, skip_final_rotation_layer=True).decompose()
            params = newProcessing.parameters

        # check if new circuit would exceed limits
        if (nLayers > maxLayers or len(params) > maxParam or newProcessing.depth() > maxDepth):
            break

        # circuit passed the check
        processing = newProcessing
        nLayers += 1
    return processing
