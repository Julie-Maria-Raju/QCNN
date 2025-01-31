import pennylane as qml
import torch
from circuitcomponents import CircuitComponents
        

class Sequence_CNOTs(CircuitComponents):
    """ Applies a sequence of CNOT gates. Per default, the gates are applied over all adjacent qubits together with
    an additional CNOT gate between the first and the last qubit. With the option `wires_to_act_on` the qubits
    involved in the sequence can be varied. Note that you can change the architecture of the CNOT sequence by
    varying the order of the qubits in `wires_to_act_on`."""

    def __init__(self, qubits, wires_to_act_on=None):
        super().__init__()
        if not wires_to_act_on:
            self.wires_to_act_on = list(range(qubits))
        elif max(wires_to_act_on) > qubits - 1:
            print("ERROR! Your chosen `wires_to_act_on` are larger than the total number of qubits. Exiting...")
            exit()
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "Sequence_CNOTs"
        self.required_qubits = max(self.wires_to_act_on)

    def circuit(self, weights):
        for i in range(len(self.wires_to_act_on) - 1):
            qml.CNOT(wires=[self.wires_to_act_on[i], self.wires_to_act_on[i + 1]])
        # connect first and last qubit for circuits with at least three qubits
        if len(self.wires_to_act_on) > 2:
            qml.CNOT(wires=[self.wires_to_act_on[-1], 0])       
        # return [qml.expval(qml.PauliZ(1))] 
    """If we want to use multirun we return [qml.expval(qml.PauliZ(1))] , and in settings measurement is None """


class Sequence_RX_CNOTs(CircuitComponents):
    """ Applies a sequence of trainable RX rotations and CNOT gates. Per default, the gates are applied over all
    adjacent qubits together with  an additional CNOT gate between the first and the last qubit. With the option
    `wires_to_act_on` the qubits involved in the sequence can be varied. Note that you can change the architecture
    of the sequence by varying the order of the qubits in `wires_to_act_on`."""

    def __init__(self, qubits, wires_to_act_on=None):
        super().__init__()
        if not wires_to_act_on:
            self.wires_to_act_on = list(range(qubits))
        elif max(wires_to_act_on) > qubits - 1:
            print("ERROR! Your chosen `wires_to_act_on` are larger than the total number of qubits. Exiting...")
            exit()
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "Sequence_RX_CNOTs"
        self.required_qubits = max(self.wires_to_act_on)

    def circuit(self, weights):
        for i in range(len(self.wires_to_act_on)):
            qml.RX(weights[i], wires=self.wires_to_act_on[i])
        for i in range(len(self.wires_to_act_on) - 1):
            qml.CNOT(wires=[self.wires_to_act_on[i], self.wires_to_act_on[i + 1]])
        # connect first and last qubit for circuits with at least three qubits
        if len(self.wires_to_act_on) > 2:
            qml.CNOT(wires=[self.wires_to_act_on[-1], 0])
        #return [qml.expval(qml.PauliZ(1))]
    """If we want to use multirun we return [qml.expval(qml.PauliZ(1))], and in settings measurement is None"""



class Sequence_RX_RY_CNOTs(CircuitComponents):
    """ Applies a sequence of trainable RX rotations and CNOT gates. Per default, the gates are applied over all
    adjacent qubits together with  an additional CNOT gate between the first and the last qubit. With the option
    `wires_to_act_on` the qubits involved in the sequence can be varied. Note that you can change the architecture
    of the sequence by varying the order of the qubits in `wires_to_act_on`."""

    def __init__(self, qubits, wires_to_act_on=None):
        super().__init__()
        if not wires_to_act_on:
            self.wires_to_act_on = list(range(qubits))
        elif max(wires_to_act_on) > qubits - 1:
            print("ERROR! Your chosen `wires_to_act_on` are larger than the total number of qubits. Exiting...")
            exit()
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "Sequence_RX_RY_CNOTs"
        self.required_qubits = max(self.wires_to_act_on)

    def circuit(self, weights):
        for i in range(len(self.wires_to_act_on)):
            qml.RX(weights[i], wires=self.wires_to_act_on[i])
            qml.RY(weights[2*i], wires=self.wires_to_act_on[i])
        for i in range(len(self.wires_to_act_on) - 1):
            qml.CNOT(wires=[self.wires_to_act_on[i], self.wires_to_act_on[i + 1]])
        # connect first and last qubit for circuits with at least three qubits
        if len(self.wires_to_act_on) > 2:
            qml.CNOT(wires=[self.wires_to_act_on[-1], 0])


class strongEnt_Rot_CRY(CircuitComponents):
    """ Applies a sequence of trainable Rot and CRY gates. Per default, the gates are applied over all
    adjacent qubits together with an additional CRY gate between the first and the last qubit. With the option
    `wires_to_act_on` the qubits involved in the sequence can be varied. Note that you can change the architecture
    of the sequence by varying the order of the qubits in `wires_to_act_on`.
    The Parameter r (range) of the Pennylane StronglyEntangledLayer is not implemented at the moment.
    The number of available weights for this quantum layer is defined in circuitcomponents_utils.py, get_circuit_weights_shape().
    """

    def __init__(self, qubits, wires_to_act_on=None):
        super().__init__()
        if not wires_to_act_on:
            self.wires_to_act_on = list(range(qubits))
        elif max(wires_to_act_on) > qubits - 1:
            print("ERROR! Your chosen `wires_to_act_on` are larger than the total number of qubits. Exiting...")
            exit()
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "strongEnt_Rot_CRY"
        self.required_qubits = max(self.wires_to_act_on)

    def circuit(self, weights):
        for i in range(len(self.wires_to_act_on)):
            qml.Rot(weights[0, i], weights[1, i], weights[2, i], wires=self.wires_to_act_on[i])
        for i in range(len(self.wires_to_act_on) - 1):
            qml.CRY(weights[3, i], wires=[self.wires_to_act_on[i], self.wires_to_act_on[i + 1]])
        # connect first and last qubit for circuits with at least three qubits
        if len(self.wires_to_act_on) > 2:
            qml.CRY(weights[3, i], wires=[self.wires_to_act_on[-1], 0])

class Sequence_ancilla_RX_CZs(CircuitComponents):
    """ This requires to put ancilla_qbit=1 and measurement=None. Applies a sequence of trainable RX rotations and CNOT gates on the normal qbits. 
    The ancilla qbit follows the H-U-H fashion."""

    def __init__(self, qubits, wires_to_act_on=None):
        super().__init__()
        if not wires_to_act_on:
            self.wires_to_act_on = list(range(qubits+1))
        elif max(wires_to_act_on) > qubits - 1:
            print("ERROR! Your chosen `wires_to_act_on` are larger than the total number of qubits. Exiting...")
            exit()
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "Sequence_ancilla_RX_CNOTs"
        self.required_qubits = max(self.wires_to_act_on)
    def circuit(self, weights):
        # basic ent
        for i in range(len(self.wires_to_act_on)-1):
            qml.RX(weights[i], wires=self.wires_to_act_on[i])
        for i in range(len(self.wires_to_act_on) - 2):
            qml.CNOT(wires=[self.wires_to_act_on[i], self.wires_to_act_on[i + 1]])
        # connect first and last qubit for circuits with at least three qubits
        if len(self.wires_to_act_on) > 2:
            qml.CNOT(wires=[self.wires_to_act_on[-2], 0])

        # ancilla
        qml.Hadamard(len(self.wires_to_act_on)-1)
        for i in range(0,len(self.wires_to_act_on) - 1):
            qml.CZ(wires=[i, self.wires_to_act_on[-1]])
        qml.Hadamard(len(self.wires_to_act_on)-1)
        return [qml.expval(qml.PauliZ(wires=max(self.wires_to_act_on)))]

class Sequence_ancilla_RX_CYs(CircuitComponents):
    """ This requires to put ancilla_qbit=1 and measurement=None. Applies a sequence of trainable RX rotations and CNOT gates on the normal qbits. 
    The ancilla qbit follows the H-U-H fashion."""

    def __init__(self, qubits, wires_to_act_on=None):
        super().__init__()
        if not wires_to_act_on:
            self.wires_to_act_on = list(range(qubits+1))
        elif max(wires_to_act_on) > qubits - 1:
            print("ERROR! Your chosen `wires_to_act_on` are larger than the total number of qubits. Exiting...")
            exit()
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "Sequence_ancilla_RX_CNOTs"
        self.required_qubits = max(self.wires_to_act_on)
    def circuit(self, weights):
        # basic ent
        for i in range(len(self.wires_to_act_on)-1):
            qml.RX(weights[i], wires=self.wires_to_act_on[i])
        for i in range(len(self.wires_to_act_on) - 2):
            qml.CNOT(wires=[self.wires_to_act_on[i], self.wires_to_act_on[i + 1]])
        # connect first and last qubit for circuits with at least three qubits
        if len(self.wires_to_act_on) > 2:
            qml.CNOT(wires=[self.wires_to_act_on[-2], 0])

        # ancilla
        qml.Hadamard(len(self.wires_to_act_on)-1)
        for i in range(0,len(self.wires_to_act_on) - 1):
            qml.CY(wires=[i, self.wires_to_act_on[-1]])
        qml.Hadamard(len(self.wires_to_act_on)-1)
        return [qml.expval(qml.PauliZ(wires=max(self.wires_to_act_on)))]

class Sequence_ancilla_RX_CNOTs(CircuitComponents):
    """ This requires to put ancilla_qbit=1 and measurement=None. Applies a sequence of trainable RX rotations and CNOT gates on the normal qbits. 
    The ancilla qbit follows the H-U-H fashion."""

    def __init__(self, qubits, wires_to_act_on=None):
        super().__init__()
        if not wires_to_act_on:
            self.wires_to_act_on = list(range(qubits+1))
        elif max(wires_to_act_on) > qubits - 1:
            print("ERROR! Your chosen `wires_to_act_on` are larger than the total number of qubits. Exiting...")
            exit()
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "Sequence_ancilla_RX_CNOTs"
        self.required_qubits = max(self.wires_to_act_on)
    def circuit(self, weights):
        for i in range(len(self.wires_to_act_on)-1):
            qml.RX(weights[i], wires=self.wires_to_act_on[i])
        for i in range(len(self.wires_to_act_on) - 2):
            qml.CNOT(wires=[self.wires_to_act_on[i], self.wires_to_act_on[i + 1]])
        qml.Hadamard(len(self.wires_to_act_on)-1)
        for i in range(0,len(self.wires_to_act_on) - 1):
            qml.CNOT(wires=[i, self.wires_to_act_on[-1]])
        qml.Hadamard(len(self.wires_to_act_on)-1)
        return [qml.expval(qml.PauliZ(wires=max(self.wires_to_act_on)))]
    """if we want ancilla wihout RX_CNOT ("pure") this is the circuit:
    def circuit(self, weights):
        qml.Hadamard(len(self.wires_to_act_on)-1)
        for i in range(1,len(self.wires_to_act_on) - 1):
            qml.CZ(wires=[self.wires_to_act_on[-1], i])
        qml.Hadamard(len(self.wires_to_act_on)-1)
        return [qml.expval(qml.PauliZ(wires=max(self.wires_to_act_on)))]"""

        

class quantumTemplate(CircuitComponents):
    """
    This is an empty template for the quantum layer where you can place your own quantum gates.
    The number of available weights for this quantum layer is defined in circuitcomponents_utils.py, get_circuit_weights_shape().
    """

    def __init__(self, qubits, wires_to_act_on=None):
        super().__init__()
        if not wires_to_act_on:
            self.wires_to_act_on = list(range(qubits))
        elif max(wires_to_act_on) > qubits - 1:
            print("ERROR! Your chosen `wires_to_act_on` are larger than the total number of qubits. Exiting...")
            exit()
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "quantumTemplate"
        self.required_qubits = max(self.wires_to_act_on)

    def circuit(self, weights):
        for i in range(len(self.wires_to_act_on)):
            pass


# place for more classes with custom entanglements
