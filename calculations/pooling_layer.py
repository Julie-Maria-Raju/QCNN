from circuitcomponents import CircuitComponents
import pennylane as qml


class PoolingLayerBuildingBlocks():
    """
    PoolingLayerBuildingBlocks includes the building blocks for the different pooling architectures
    Highly inspired by http://dx.doi.org/10.1007/s42484-021-00061-x

    """
    
    @staticmethod
    def conv_ansatz(params, wires):
        qml.RY(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])

    @staticmethod
    def pooling_ansatz(params, wires):
        qml.CRZ(params[0], wires=[wires[0], wires[1]])
        qml.PauliX(wires=wires[0])
        qml.CRX(params[1], wires=[wires[0], wires[1]])
    
    @staticmethod
    def conv_ansatz_had(params, wires):
        qml.Hadamard(wires[0])
        qml.Hadamard(wires[1])
        qml.CZ(wires=[wires[0], wires[1]])
        qml.RX(params[0], wires=wires[0])
        qml.RX(params[1], wires=wires[1])


class PoolingLayer(CircuitComponents, PoolingLayerBuildingBlocks):
    
    def __init__(self, qubits, seed=0, wires_to_act_on=None):
        CircuitComponents().__init__()
        self.seed = seed

        if not wires_to_act_on or len(wires_to_act_on) > qubits:
            self.wires_to_act_on = list(range(qubits))
        else:
            self.wires_to_act_on = wires_to_act_on
        self.required_qubits = max(self.wires_to_act_on)

        self.name = "PoolingLayer"

    def circuit(self, weights):

        self.pooling_ansatz(params=[weights[0, 0], weights[0, 1]], wires=[self.wires_to_act_on[0], self.wires_to_act_on[1]])
        self.pooling_ansatz(params=[weights[1, 0], weights[1, 1]], wires=[self.wires_to_act_on[2], self.wires_to_act_on[3]])

        self.pooling_ansatz(params=[weights[2, 0], weights[2, 1]], wires=[self.wires_to_act_on[1], self.wires_to_act_on[3]])

        return [qml.expval(qml.PauliZ(wires=max(self.wires_to_act_on)))]
    
class ConvPoolingLayer(CircuitComponents, PoolingLayerBuildingBlocks):

    def __init__(self, qubits, seed=0, wires_to_act_on=None):
        CircuitComponents().__init__()
        self.seed = seed #not needed, problem if removed?

        if not wires_to_act_on or len(wires_to_act_on) > qubits:
            self.wires_to_act_on = list(range(qubits))
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "ConvPoolingLayer"
        self.required_qubits = max(self.wires_to_act_on)

    def circuit(self, weights):
        self.conv_ansatz(params=[weights[0, 0], weights[0, 1]], wires=[self.wires_to_act_on[0], self.wires_to_act_on[1]])
        self.conv_ansatz(params=[weights[1, 0], weights[1, 1]], wires=[self.wires_to_act_on[2], self.wires_to_act_on[3]])

        self.pooling_ansatz(params=[weights[0, 2], weights[0, 3]], wires=[self.wires_to_act_on[0], self.wires_to_act_on[1]])
        self.pooling_ansatz(params=[weights[1, 2], weights[1, 3]], wires=[self.wires_to_act_on[2], self.wires_to_act_on[3]])

        self.conv_ansatz(params=[weights[2, 0], weights[2, 1]], wires=[self.wires_to_act_on[1], self.wires_to_act_on[3]])
        self.pooling_ansatz(params=[weights[2, 2], weights[2, 3]], wires=[self.wires_to_act_on[1], self.wires_to_act_on[3]])

        return [qml.expval(qml.PauliZ(wires=max(self.wires_to_act_on)))]



class ConvPoolingLayerHad(CircuitComponents, PoolingLayerBuildingBlocks):

    def __init__(self, qubits, seed=0, wires_to_act_on=None):
        CircuitComponents().__init__()
        self.seed = seed #not needed, problem if removed?

        if not wires_to_act_on or len(wires_to_act_on) > qubits:
            self.wires_to_act_on = list(range(qubits))
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "ConvPoolingLayerHad"
        self.required_qubits = max(self.wires_to_act_on)

    def circuit(self, weights):
        
        self.conv_ansatz_had(params=[weights[0, 0], weights[0, 1]], wires=[self.wires_to_act_on[0], self.wires_to_act_on[1]])
        self.conv_ansatz_had(params=[weights[1, 0], weights[1, 1]], wires=[self.wires_to_act_on[2], self.wires_to_act_on[3]])

        self.pooling_ansatz(params=[weights[0, 2], weights[0, 3]], wires=[self.wires_to_act_on[0], self.wires_to_act_on[1]])
        self.pooling_ansatz(params=[weights[1, 2], weights[1, 3]], wires=[self.wires_to_act_on[2], self.wires_to_act_on[3]])

        self.conv_ansatz_had(params=[weights[2, 0], weights[2, 1]], wires=[self.wires_to_act_on[1], self.wires_to_act_on[3]])
        self.pooling_ansatz(params=[weights[2, 2], weights[2, 3]], wires=[self.wires_to_act_on[1], self.wires_to_act_on[3]])

        return [qml.expval(qml.PauliZ(wires=max(self.wires_to_act_on)))]



class ConvPoolingLayerBounded(CircuitComponents, PoolingLayerBuildingBlocks):

    def __init__(self, qubits, seed=0, wires_to_act_on=None):
        CircuitComponents().__init__()
        self.seed = seed #not needed, problem if removed?

        if not wires_to_act_on or len(wires_to_act_on) > qubits:
            self.wires_to_act_on = list(range(qubits))
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "ConvPoolingLayerBounded"
        self.required_qubits = max(self.wires_to_act_on)


    def circuit(self, weights):
        self.conv_ansatz(params=[weights[0, 0], weights[0, 1]], wires=[self.wires_to_act_on[0], self.wires_to_act_on[1]])
        self.conv_ansatz(params=[weights[1, 0], weights[1, 1]], wires=[self.wires_to_act_on[2], self.wires_to_act_on[3]])

        self.conv_ansatz(params=[weights[3, 0], weights[3, 1]], wires=[self.wires_to_act_on[1], self.wires_to_act_on[2]])
        self.conv_ansatz(params=[weights[3, 2], weights[3, 3]], wires=[self.wires_to_act_on[3], self.wires_to_act_on[0]])

        self.pooling_ansatz(params=[weights[0, 2], weights[0, 3]], wires=[self.wires_to_act_on[0], self.wires_to_act_on[1]])
        self.pooling_ansatz(params=[weights[1, 2], weights[1, 3]], wires=[self.wires_to_act_on[2], self.wires_to_act_on[3]])

        self.conv_ansatz(params=[weights[2, 0], weights[2, 1]], wires=[self.wires_to_act_on[1], self.wires_to_act_on[3]])
        self.pooling_ansatz(params=[weights[2, 2], weights[2, 3]], wires=[self.wires_to_act_on[1], self.wires_to_act_on[3]])

        return [qml.expval(qml.PauliZ(wires=max(self.wires_to_act_on)))]
