from pennylane import numpy as np
import pennylane as qml
from circuitcomponents import CircuitComponents


class Threshold_Encoder(CircuitComponents):
    """ Encode the image very naively. Each pixel is assigned to a qubit that is rotated by pi
        along the x axis depending on the value being under the threshold or not.
        Input is expected to be tensor or array with (filter_length * filter_length) entries.
    """

    def __init__(self, filter_length=2, threshold=0, data_3D=True):
        self.height = filter_length
        self.length = filter_length
        if data_3D:
            self.depth = filter_length
        else:
            self.depth = 1
        self.threshold = threshold
        self.name = "Threshold_Encoder"
        self.required_qubits = self.height * self.length * self.depth

    def circuit(self, image=np.array([])):
        flat_image = image.flatten()
        for i in range(self.height * self.length * self.depth):
            x = int(flat_image[i] > self.threshold)
            qml.RX(x * np.pi, wires=i)
