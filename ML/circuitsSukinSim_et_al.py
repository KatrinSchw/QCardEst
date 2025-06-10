"""
This code implements the 19 circuits from

Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms

https://arxiv.org/abs/1905.10876

Implemented by Jonas Thomsen
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def fromNameSukim(str: str, depth: int) -> tuple[QuantumCircuit, ParameterVector]:
    if str == "ansatzSukim1": return create_ansatz1(depth)
    if str == "ansatzSukim2": return create_ansatz2(depth)
    if str == "ansatzSukim3": return create_ansatz3(depth)
    if str == "ansatzSukim4": return create_ansatz4(depth)
    if str == "ansatzSukim5": return create_ansatz5(depth)
    if str == "ansatzSukim6": return create_ansatz6(depth)
    if str == "ansatzSukim7": return create_ansatz7(depth)
    if str == "ansatzSukim8": return create_ansatz8(depth)
    if str == "ansatzSukim9": return create_ansatz9(depth)
    if str == "ansatzSukim10": return create_ansatz10(depth)
    if str == "ansatzSukim11": return create_ansatz11(depth)
    if str == "ansatzSukim12": return create_ansatz12(depth)
    if str == "ansatzSukim13": return create_ansatz13(depth)
    if str == "ansatzSukim14": return create_ansatz14(depth)
    if str == "ansatzSukim15": return create_ansatz15(depth)
    if str == "ansatzSukim16": return create_ansatz16(depth)
    if str == "ansatzSukim17": return create_ansatz17(depth)
    if str == "ansatzSukim18": return create_ansatz18(depth)
    if str == "ansatzSukim19": return create_ansatz19(depth)
    raise ValueError("Circuit " + str + " is undefined")


# circuit 1 definition
def create_ansatz1(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 8 * depth)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 8 * d], qubit)
            vc.rz(params[qubit + 8 * d + 4], qubit)

    return vc, params


# circuit 2 definition
def create_ansatz2(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 8 * depth)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 8 * d], qubit)
            vc.rz(params[qubit + 8 * d + 4], qubit)

        vc.cx(3, 2)
        vc.cx(2, 1)
        vc.cx(1, 0)

    return vc, params


# circuit 3 definition
def create_ansatz3(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 11 * depth)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 11 * d], qubit)
            vc.rz(params[qubit + 11 * d + 4], qubit)

        vc.crz(params[8 + 11 * d], 3, 2)
        vc.crz(params[9 + 11 * d], 2, 1)
        vc.crz(params[10 + 11 * d], 1, 0)

    return vc, params


# circuit 4 definition
def create_ansatz4(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 11 * depth)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 11 * d], qubit)
            vc.rz(params[qubit + 11 * d + 4], qubit)

        vc.crx(params[8 + 11 * d], 3, 2)
        vc.crx(params[9 + 11 * d], 2, 1)
        vc.crx(params[10 + 11 * d], 1, 0)

    return vc, params


# circuit 5 definition
def create_ansatz5(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 28 * depth)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 28 * d], qubit)
            vc.rz(params[qubit + 28 * d + 4], qubit)

        i = 8 + 28 * d
        for qubit1 in range(3, -1, -1):
            for qubit2 in range(4):
                if qubit1 == qubit2:
                    continue
                vc.crz(params[i], qubit1, qubit2)
                i += 1

        for qubit in range(4):
            vc.rx(params[qubit + 20 + 28 * d], qubit)
            vc.rz(params[qubit + 20 + 28 * d + 4], qubit)

    return vc, params


# circuit 6 definition
def create_ansatz6(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 28 * depth)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 28 * d], qubit)
            vc.rz(params[qubit + 28 * d + 4], qubit)

        i = 8 + 28 * d
        for qubit1 in range(3, -1, -1):
            for qubit2 in range(4):
                if qubit1 == qubit2:
                    continue
                vc.crx(params[i], qubit1, qubit2)
                i += 1

        for qubit in range(4):
            vc.rx(params[qubit + 20 + 28 * d], qubit)
            vc.rz(params[qubit + 20 + 28 * d + 4], qubit)

    return vc, params


# circuit 7 definition
def create_ansatz7(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 19 * depth)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 19 * d], qubit)
            vc.rz(params[qubit + 19 * d + 4], qubit)

        vc.crz(params[8 + 19 * d], 3, 2)
        vc.crz(params[9 + 19 * d], 1, 0)

        for qubit in range(4):
            vc.rx(params[10 + qubit + 19 * d], qubit)
            vc.rz(params[10 + qubit + 19 * d + 4], qubit)

        vc.crz(params[18 + 19 * d], 2, 1)

    return vc, params


# circuit 8 definition
def create_ansatz8(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 19 * depth)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 19 * d], qubit)
            vc.rz(params[qubit + 19 * d + 4], qubit)

        vc.crx(params[8 + 19 * d], 3, 2)
        vc.crx(params[9 + 19 * d], 1, 0)

        for qubit in range(4):
            vc.rx(params[10 + qubit + 19 * d], qubit)
            vc.rz(params[10 + qubit + 19 * d + 4], qubit)

        vc.crx(params[18 + 19 * d], 2, 1)

    return vc, params


# circuit 9 definition
def create_ansatz9(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 4 * depth)

    for d in range(depth):
        for qubit in range(4):
            vc.h(qubit)

        vc.cz(3, 2)
        vc.cz(2, 1)
        vc.cz(1, 0)

        for qubit in range(4):
            vc.rx(params[qubit + 4 * d], qubit)

    return vc, params


# circuit 10 definition
def create_ansatz10(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 4 + 4 * depth)

    for qubit in range(4):
        vc.ry(params[qubit], qubit)

    for d in range(depth):

        vc.cz(3, 2)
        vc.cz(2, 1)
        vc.cz(1, 0)
        vc.cz(0, 3)

        for qubit in range(4):
            vc.ry(params[4 + qubit + 4 * d], qubit)

    return vc, params


# circuit 11 definition
def create_ansatz11(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 12 * depth)

    for d in range(depth):
        for qubit in range(4):
            vc.ry(params[qubit + 12 * d], qubit)
            vc.rz(params[qubit + 12 * d + 4], qubit)

        vc.cx(3, 2)
        vc.cx(1, 0)

        vc.ry(params[8 + 12 * d], 1)
        vc.ry(params[9 + 12 * d], 2)
        vc.rz(params[10 + 12 * d], 1)
        vc.rz(params[11 + 12 * d], 2)

        vc.cx(2, 1)

    return vc, params


# circuit 12 definition
def create_ansatz12(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 12 * depth)
    param_iter = iter(params)

    for d in range(depth):
        for qubit in range(4):
            vc.ry(params[qubit + 12 * d], qubit)
            vc.rz(params[qubit + 12 * d + 4], qubit)

        vc.cz(3, 2)
        vc.cz(1, 0)

        vc.ry(params[8 + 12 * d], 1)
        vc.ry(params[9 + 12 * d], 2)
        vc.rz(params[10 + 12 * d], 1)
        vc.rz(params[11 + 12 * d], 2)

        vc.cz(2, 1)

    return vc, params


# circuit 13 definition
def create_ansatz13(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 16 * depth)
    param_iter = iter(params)

    for d in range(depth):
        for qubit in range(4):
            vc.ry(params[qubit + 16 * d], qubit)

        for qubit in range(4):
            vc.crz(params[qubit + 16 * d + 4], 3 - qubit, (4 - qubit) % 4)

        for qubit in range(4):
            vc.ry(params[qubit + 16 * d + 8], qubit)

        for qubit in range(4):
            vc.crz(params[qubit + 16 * d + 12], (3 + qubit) % 4, (2 + qubit) % 4)

    return vc, params


# circuit 14 definition
def create_ansatz14(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 16 * depth)
    param_iter = iter(params)

    for d in range(depth):
        for qubit in range(4):
            vc.ry(params[qubit + 16 * d], qubit)

        for qubit in range(4):
            vc.crx(params[qubit + 16 * d + 4], 3 - qubit, (4 - qubit) % 4)

        for qubit in range(4):
            vc.ry(params[qubit + 16 * d + 8], qubit)

        for qubit in range(4):
            vc.crx(params[qubit + 16 * d + 12], (3 + qubit) % 4, (2 + qubit) % 4)

    return vc, params


# circuit 15 definition
def create_ansatz15(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 8 * depth)
    param_iter = iter(params)

    for d in range(depth):
        for qubit in range(4):
            vc.ry(params[qubit + 8 * d], qubit)

        for qubit in range(4):
            vc.cx(3 - qubit, (4 - qubit) % 4)

        for qubit in range(4):
            vc.ry(params[qubit + 8 * d + 4], qubit)

        for qubit in range(4):
            vc.cx((3 + qubit) % 4, (2 + qubit) % 4)

    return vc, params


# circuit 16 definition
def create_ansatz16(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 11 * depth)
    param_iter = iter(params)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 11 * d], qubit)
            vc.rz(params[qubit + 11 * d + 4], qubit)

        vc.crz(params[8 + 11 * d], 3, 2)
        vc.crz(params[9 + 11 * d], 1, 0)
        vc.crz(params[10 + 11 * d], 2, 1)

    return vc, params


# circuit 17 definition
def create_ansatz17(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 11 * depth)
    param_iter = iter(params)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 11 * d], qubit)
            vc.rz(params[qubit + 11 * d + 4], qubit)

        vc.crx(params[8 + 11 * d], 3, 2)
        vc.crx(params[9 + 11 * d], 1, 0)
        vc.crx(params[10 + 11 * d], 2, 1)

    return vc, params


# circuit 18 definition
def create_ansatz18(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 8 * depth)
    param_iter = iter(params)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 8 * d], qubit)
            vc.rz(params[qubit + 8 * d + 4], qubit)

        for qubit in range(4):
            vc.cz(3 - qubit, (4 - qubit) % 4)

    return vc, params


# circuit 19 definition
def create_ansatz19(depth=1):
    vc = QuantumCircuit(4)
    params = ParameterVector('theta', 8 * depth)
    param_iter = iter(params)

    for d in range(depth):
        for qubit in range(4):
            vc.rx(params[qubit + 8 * d], qubit)
            vc.rz(params[qubit + 8 * d + 4], qubit)

        for qubit in range(4):
            vc.cx(3 - qubit, (4 - qubit) % 4)

    return vc, params
