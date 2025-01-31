from pennylane import numpy as np
import pennylane as qml
from circuitcomponents import CircuitComponents
import math


class Custom_Higher_Order_Encoder(CircuitComponents):
    """ 
        Encode the image following the higher order encoding idea presented in https://arxiv.org/abs/2011.00027v1
	The idea is to first add every value to a rotation gate at every qubit and then to connect two qubits by a product of two input values.
        Input is expected to be tensor or array with (filter_length * filter_length) entries.
    """

    def __init__(self, filter_length=2, data_3D=True, rotation_factor=math.pi):
        self.height = filter_length
        self.length = filter_length
        if data_3D:
            self.depth = filter_length
        else:
            self.depth = 1
        self.name = "Custom_Higher_Order_Encoder"
        self.required_qubits = self.height * self.length * self.depth
        self.rotation_factor = rotation_factor

    def circuit(self, image=np.array([])):
        flat_image = image.flatten()
        for i in range(self.required_qubits):
            qml.Hadamard(i)
            qml.RZ(flat_image[i]*self.rotation_factor, wires=i)
        for i in range(self.required_qubits - 1):
            for j in range(i+1, self.required_qubits):
                feature_product = flat_image[i]*flat_image[j]*self.rotation_factor
                qml.CNOT(wires=[i, j])
                qml.RZ(feature_product, wires=j)
                qml.CNOT(wires=[i, j])
