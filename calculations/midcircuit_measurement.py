import pennylane as qml
import torch
from circuitcomponents import CircuitComponents

class Sequence_mid_measure(CircuitComponents):
    """ Implements a quantum circuit that dynamically changes its behavior based on measurement results. 
    The sequence of RX rotations depends on the measurement outcome of a qbit whereas CNOT gates are applied in all cases"""

    def __init__(self, qubits, wires_to_act_on=None):
        super().__init__()
        if not wires_to_act_on:
            self.wires_to_act_on = list(range(qubits))
        elif max(wires_to_act_on) > qubits - 1:
            print("ERROR! Your chosen `wires_to_act_on` are larger than the total number of qubits. Exiting...")
            exit()
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "Sequence_mid_measure"
        self.required_qubits = max(self.wires_to_act_on)
        
    def circuit(self, weights):
        m_0=qml.measure(self.wires_to_act_on[0])
        for i in range(1, len(self.wires_to_act_on)):
            qml.cond(m_0, qml.RX)(weights[i-1], wires=self.wires_to_act_on[i])
        m_1=qml.measure(self.wires_to_act_on[1])
        for i in range(2, len(self.wires_to_act_on)):
            qml.cond(m_1, qml.RX)(weights[i+1], wires=self.wires_to_act_on[i])
        for i in range(2, len(self.wires_to_act_on)-1):
            qml.CNOT(wires=[self.wires_to_act_on[i], self.wires_to_act_on[i+1]])
        m_2=qml.measure(self.wires_to_act_on[2])
        for i in range(3, len(self.wires_to_act_on)):
            qml.cond(m_2, qml.RX)(weights[i+2], wires=self.wires_to_act_on[i])
        return [qml.expval(qml.PauliZ(wires=max(self.wires_to_act_on)))]