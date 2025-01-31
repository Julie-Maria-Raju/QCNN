from pennylane import numpy as np
import pennylane as qml
from circuitcomponents import CircuitComponents

class Amplitude_Encoder(CircuitComponents):
    """ 
    Data is encoded into the amplitudes of a quantum state.
    """

    def __init__(self, filter_length=2, data_3D=True):
        self.height = filter_length
        self.length = filter_length
        if data_3D:
            self.depth = filter_length
        else:
            self.depth = 1
        self.name = "Amplitude_Encoder"
        self.required_qubits = int(np.log2(self.height * self.length * self.depth))

    def circuit(self, image=np.array([])):
        flat_image = image.flatten()
        qml.AmplitudeEmbedding(features=flat_image, wires=range(self.required_qubits), normalize=True)
        