import pennylane as qml
from circuitcomponents import CircuitComponents


class CharacterizedCircuits():
    '''These are the circuits from the Sim et al paper https://arxiv.org/abs/1905.10876.'''

    def __init__(self, qubits, wires_to_act_on=None): #TODO: how to add layers?
        if not wires_to_act_on:
            self.wires_to_act_on = list(range(qubits))
        elif max(wires_to_act_on) > qubits - 1:
            print("ERROR! Your chosen `wires_to_act_on` are larger than the total number of qubits. Exiting...")
            exit()
        else:
            self.wires_to_act_on = wires_to_act_on
        self.required_qubits = max(self.wires_to_act_on)


class Circuit_2(CircuitComponents):
    def __init__(self, qubits, wires_to_act_on=None):
        super(Circuit_2, self).__init__()
        CharacterizedCircuits.__init__(self, qubits, wires_to_act_on)

    def circuit(self, weights):
        weight_index = 0
        for i in range(len(self.wires_to_act_on)):
            qml.RX(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1

            qml.RZ(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1
                    
        for cntrl_wire_index in range(len(self.wires_to_act_on)-1, 0, -1):
            qml.CNOT(wires=(cntrl_wire_index, cntrl_wire_index-1))


class Circuit_3(CircuitComponents):
    def __init__(self, qubits, wires_to_act_on=None):
        super(Circuit_3, self).__init__()
        CharacterizedCircuits.__init__(self, qubits, wires_to_act_on)

    def circuit(self, weights):
        weight_index = 0
        for i in range(len(self.wires_to_act_on)):
            qml.RX(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1

            qml.RZ(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1
                    
        for cntrl_wire_index in range(len(self.wires_to_act_on)-1, 0, -1):
            qml.CRZ(weights[weight_index], wires=(cntrl_wire_index, cntrl_wire_index-1))
            weight_index += 1


class Circuit_4(CircuitComponents):
    def __init__(self, qubits, wires_to_act_on=None):
        super(Circuit_4, self).__init__()
        CharacterizedCircuits.__init__(self, qubits, wires_to_act_on)

    def circuit(self, weights):
        weight_index = 0
        for i in range(len(self.wires_to_act_on)):
            qml.RX(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1

            qml.RZ(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1
                    
        for cntrl_wire_index in range(len(self.wires_to_act_on)-1, 0, -1):
            qml.CRX(weights[weight_index], wires=(cntrl_wire_index, cntrl_wire_index-1))
            weight_index += 1

class Circuit_5(CircuitComponents):
    def __init__(self, qubits, wires_to_act_on=None):
        super(Circuit_5, self).__init__()
        CharacterizedCircuits.__init__(self, qubits, wires_to_act_on)

    def circuit(self, weights):
        weight_index = 0
        for i in range(len(self.wires_to_act_on)):
            qml.RX(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1

            qml.RZ(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1
        qml.Barrier()
                    
        for cntrl_wire_index in range(len(self.wires_to_act_on)-1, -1, -1):
            for target_wire_index in range(len(self.wires_to_act_on)-1, -1, -1):
                if cntrl_wire_index != target_wire_index:
                    qml.CRZ(weights[weight_index], wires = (cntrl_wire_index, target_wire_index))
                    weight_index += 1
        qml.Barrier()

        for i in range(len(self.wires_to_act_on)):
            qml.RX(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1

            qml.RZ(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1


class Circuit_6(CircuitComponents):
    def __init__(self, qubits, wires_to_act_on=None):
        super(Circuit_6, self).__init__()
        CharacterizedCircuits.__init__(self, qubits, wires_to_act_on)

    def circuit(self, weights):
        weight_index = 0
        for i in range(len(self.wires_to_act_on)):
            qml.RX(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1

            qml.RZ(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1
        qml.Barrier()

        for cntrl_wire_index in range(len(self.wires_to_act_on)-1, -1, -1):
            for target_wire_index in range(len(self.wires_to_act_on)-1, -1, -1):
                if cntrl_wire_index != target_wire_index:
                    qml.CRX(weights[weight_index], wires = (cntrl_wire_index, target_wire_index))
                    weight_index += 1
        qml.Barrier()

        for i in range(len(self.wires_to_act_on)):
            qml.RX(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1

            qml.RZ(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1


class Circuit_10(CircuitComponents):
    def __init__(self, qubits, wires_to_act_on=None):
        super(Circuit_10, self).__init__()
        CharacterizedCircuits.__init__(self, qubits, wires_to_act_on)

    def circuit(self, weights):
        weight_index = 0
        for i in range(len(self.wires_to_act_on)):
            qml.RY(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1

        for i in range(len(self.wires_to_act_on)-1, 0, -1):
            qml.CZ(wires=(i, i - 1))
            
        qml.CZ(wires=(0, len(self.wires_to_act_on)-1))
            
        for i in range(len(self.wires_to_act_on)):
            qml.RY(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1


class Circuit_14(CircuitComponents):
    def __init__(self, qubits, wires_to_act_on=None):
        super(Circuit_14, self).__init__()
        CharacterizedCircuits.__init__(self, qubits, wires_to_act_on)

    def circuit(self, weights):
        weight_index = 0
        for i in range(len(self.wires_to_act_on)):
            qml.RY(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1
            
        for cntrl_wire_index in range(len(self.wires_to_act_on)-1, -1, -1):
            qml.CRX(weights[weight_index], wires = (cntrl_wire_index, (cntrl_wire_index + 1) % len(self.wires_to_act_on)))
            weight_index += 1

        for i in range(len(self.wires_to_act_on)):
            qml.RY(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1
            
        for i in [len(self.wires_to_act_on) - 1] + list(range(len(self.wires_to_act_on) - 1)):
            if i > 0:
                qml.CRX(weights[weight_index], wires = (self.wires_to_act_on[i], self.wires_to_act_on[i] - 1))
            else:
                qml.CRX(weights[weight_index], wires = (0, len(self.wires_to_act_on) - 1))
            weight_index += 1


class Circuit_19(CircuitComponents):
    def __init__(self, qubits, wires_to_act_on=None):
        super(Circuit_19, self).__init__()
        CharacterizedCircuits.__init__(self, qubits, wires_to_act_on)

    def circuit(self, weights):
        weight_index = 0
        for i in range(len(self.wires_to_act_on)):
            qml.RX(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1
        
        for i in range(len(self.wires_to_act_on)):
            qml.RZ(weights[weight_index], wires=self.wires_to_act_on[i])
            weight_index += 1

        for cntrl_wire_index in range(len(self.wires_to_act_on)-1, -1, -1):
            qml.CRX(weights[weight_index], wires = (cntrl_wire_index, (cntrl_wire_index + 1) % len(self.wires_to_act_on)))
            weight_index += 1



