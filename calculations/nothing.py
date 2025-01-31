from circuitcomponents import CircuitComponents


class Nothing(CircuitComponents):
    """ Empty calculation layer. """

    def __init__(self, qubits):
        self.name = "Nothing"
        self.required_qubits = 0

    def circuit(self, weights):
        pass
