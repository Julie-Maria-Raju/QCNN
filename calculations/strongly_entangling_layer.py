from circuitcomponents import CircuitComponents
import pennylane as qml


class StronglyEntanglingLayer(CircuitComponents):
    ''' Layers consisting of single qubit rotations and entanglers, inspired by the circuit-centric classifier design
    `arXiv:1804.00633. Directly from qml.templates.StronglyEntanglingLayers.'''

    def __init__(self, qubits, seed=0, wires_to_act_on=None):
        super().__init__()
        self.seed = seed
        if not wires_to_act_on or len(wires_to_act_on) > qubits:
            self.wires_to_act_on = list(range(qubits))
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "StronglyEntanglingLayer"
        self.required_qubits = max(self.wires_to_act_on)

    def circuit(self, weights):
        '''
        ranges: sequence determining the range hyperparameter for each subsequent layer; if None using
        r = l mod M  for the l-th layer and M wires
        imprimitive: two-qubit gate to use, defaults to CNOT
        '''
        qml.templates.StronglyEntanglingLayers(
            weights.unsqueeze(0), wires=self.wires_to_act_on, ranges=None, imprimitive=None)


class StronglyEntanglingLayerHadamard(CircuitComponents):
    ''' Layers consisting of single qubit rotations and entanglers, inspired by the circuit-centric classifier design
    `arXiv:1804.00633. Directly from qml.templates.StronglyEntanglingLayers.'''

    def __init__(self, qubits, seed=0, wires_to_act_on=None):
        super().__init__()
        self.seed = seed
        if not wires_to_act_on or len(wires_to_act_on) > qubits:
            self.wires_to_act_on = list(range(qubits))
        else:
            self.wires_to_act_on = wires_to_act_on
        self.name = "StronglyEntanglingLayer"
        self.required_qubits = max(self.wires_to_act_on)

    def circuit(self, weights):
        '''
        ranges: sequence determining the range hyperparameter for each subsequent layer; if None using
        r = l mod M  for the l-th layer and M wires
        imprimitive: two-qubit gate to use, defaults to CNOT
        '''
        qml.templates.StronglyEntanglingLayers(
            weights.unsqueeze(0), wires=self.wires_to_act_on, ranges=None, imprimitive=None)
        for i in range(self.required_qubits + 1):
            qml.Hadamard(i)
