from pennylane import numpy as np
import pennylane as qml
import math
from circuitcomponents import CircuitComponents

class Angle_Encoder(CircuitComponents):
    """ 
        Encodes N features into the rotation angles of n qubits, where N ≤ n.
        The rotations can be chosen as either RX, RY or RZ gates, as defined by the rotation parameter.
    """

    def __init__(self, filter_length=2, data_3D=True):
        self.height = filter_length
        self.length = filter_length
        if data_3D:
            self.depth = filter_length
        else:
            self.depth = 1
        self.name = "Angle_Encoder"
        self.required_qubits = self.height * self.length * self.depth

    def circuit(self, image=np.array([])):
        flat_image = image.flatten()
        qml.AngleEmbedding(flat_image*math.pi, wires=range(self.height * self.length * self.depth), rotation='X')

