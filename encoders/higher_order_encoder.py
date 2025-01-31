from pennylane import numpy as np
import pennylane as qml
from circuitcomponents import CircuitComponents


class Higher_Order_Encoder(CircuitComponents):
    """ 
        Encode the image following the higher order encoding idea presented in https://arxiv.org/abs/2011.00027v1
	The idea is to first add every value to a rotation gate at every qubit and then to connect two qubits by a product of two input values.
        Input is expected to be tensor or array with (filter_length * filter_length) entries.
    """

    def __init__(self, filter_length=2, data_3D=True):
        self.height = filter_length
        self.length = filter_length
        if data_3D:
            self.depth = filter_length
        else:
            self.depth = 1
        self.name = "Higher_Order_Encoder"
        self.required_qubits = self.height * self.length * self.depth

    def circuit(self, image=np.array([])):
        flat_image = image.flatten()
        qml.IQPEmbedding(flat_image, wires=range(self.height * self.length * self.depth))

